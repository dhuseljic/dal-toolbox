import math
import copy
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import ood, generalization


class SNGP(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 in_features: int,
                 num_classes: int,
                 num_inducing: int = 1024,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 scale_random_features: bool = False,
                 mean_field_factor: float = 1,
                 cov_momentum: float = .999,
                 ridge_penalty: float = 1e-6,
                 ):
        super().__init__()
        self.model = model

        # in_features -> num_inducing -> num_classes
        self.in_features = in_features
        self.num_inducing = num_inducing
        self.num_classes = num_classes

        # Scale input
        self.kernel_scale = kernel_scale

        # Norm input
        self.normalize_input = normalize_input
        if self.normalize_input:
            self.kernel_scale = 1
            self.layer_norm = nn.LayerNorm(in_features)

        # Random features
        self.scale_random_features = scale_random_features

        # Inference
        self.mean_field_factor = mean_field_factor

        # Covariance computation
        self.cov_momentum = cov_momentum
        self.ridge_penalty = ridge_penalty

        self.random_features = RandomFourierFeatures(
            in_features=in_features,
            num_inducing=num_inducing,
            kernel_scale=self.kernel_scale,
            scale_features=self.scale_random_features
        )

        # Define output layer according to Eq 8., For imagenet init with normal std=0.01
        self.beta = nn.Linear(num_inducing, num_classes, bias=False)
        nn.init.xavier_normal_(self.beta.weight)

        # precision matrix
        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)

        self.sampled_betas = None

    def reset_covariance(self):
        device = self.precision_matrix.device
        self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)
        self.precision_matrix.to(device)

    def update_precision_matrix(self, phi):
        precision_matrix_minibatch = torch.matmul(phi.T, phi)
        if self.cov_momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        self.precision_matrix = nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, phi):
        covariance_matrix_feature = torch.linalg.pinv(self.precision_matrix)
        out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(phi, out)
        return covariance_matrix_gp

    def forward(self, x, return_cov=False, update_precision=True):
        _, features = self.model(x, return_features=True)

        if self.normalize_input:
            features = self.layer_norm(features)

        # Get gp features according to Eq. 7
        phi = self.random_features(features)

        # Eq. 8
        logits = self.beta(phi)

        if update_precision:
            self.update_precision_matrix(phi)
        if return_cov:
            cov = self.compute_predictive_covariance(phi)
            return logits, cov
        return logits

    @torch.no_grad()
    def forward_mean_field(self, x):
        logits, cov = self.forward(x, return_cov=True, update_precision=False)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits

    @torch.no_grad()
    def forward_sample(self, x, n_draws=10, resample=False, return_dist=False):
        if self.sampled_betas is None or resample:
            # Dist from the normal that approximates the posterior over beta
            dist = torch.distributions.MultivariateNormal(
                loc=self.beta.weight,
                precision_matrix=self.precision_matrix
            )
            self.sampled_betas = dist.sample(sample_shape=(n_draws,))
            # TODO return_dist

        _, features = self.model(x, return_features=True)
        if self.normalize_input:
            features = self.layer_norm(features)
        phi = self.random_features(features)
        logits_sampled = torch.einsum('nd,ekd->enk', phi, self.sampled_betas)
        return logits_sampled

    @torch.no_grad()
    def compute_weights(self, x, y, n_draws=10):
        _, features = self.model(x, return_features=True)

        if self.normalize_input:
            features = self.layer_norm(features)
        phi = self.random_features(features)

        # Dist from the normal that approximates the posterior over beta
        dist = torch.distributions.MultivariateNormal(
            loc=self.beta.weight,
            precision_matrix=self.precision_matrix
        )
        sampled_betas = dist.sample(sample_shape=(n_draws,))
        logits_sampled = torch.einsum('nd,ekd->enk', phi, sampled_betas)
        probas_sampled = logits_sampled.sotmax(-1)

        return logits_sampled


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, num_inducing=1024, kernel_scale=1, scale_features=True):
        super().__init__()
        self.kernel_scale = kernel_scale
        self.input_scale = 1/math.sqrt(self.kernel_scale)

        self.scale_features = scale_features
        self.random_feature_scale = math.sqrt(2./float(num_inducing))

        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
        # TODO: change init for 2d?
        nn.init.normal_(self.random_feature_linear.weight, std=.05)
        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

    def forward(self, x):
        # Supports lengthscale for cutom random feature layer by directly rescaling the input.
        x = x * self.input_scale
        x = torch.cos(self.random_feature_linear(x))

        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/wide_resnet_sngp.py#L207
        if self.scale_features:
            # Scale random feature by 2. / sqrt(num_inducing).  When using GP
            # layer as the output layer of a nerual network, it is recommended
            # to turn this scaling off to prevent it from changing the learning
            # rate to the hidden layers.
            x = self.random_feature_scale * x
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.reset_covariance()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        acc1, = generalization.accuracy(outputs, targets, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


def mean_field_logits(logits, cov, lmb=math.pi / 8):
    """Scale logits using the mean field approximation proposed by https://arxiv.org/abs/2006.07584"""
    if lmb is None or lmb < 0:
        return logits
    variances = torch.diag(cov).view(-1, 1) if cov is not None else 1
    logits_adjusted = logits / torch.sqrt(1 + lmb*variances)
    return logits_adjusted


@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    test_stats = {}
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_scaled = model.forward_mean_field(inputs)
        logits_id.append(logits_scaled)
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Update test stats
    loss = criterion(logits_id, targets_id)
    acc1, = generalization.accuracy(logits_id, targets_id, (1,))
    test_stats.update({
        'loss': loss.item(),
        'acc1': acc1.item()
    })

    # Forward prop out of distribution
    logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_scaled = model.forward_mean_field(inputs)
        logits_ood.append(logits_scaled)
    logits_ood = torch.cat(logits_ood, dim=0).cpu()

    # Update test stats
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    test_stats.update({'auroc': ood.ood_auroc(entropy_id, entropy_ood)})
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood.ood_auroc(1-conf_id, 1-conf_ood)})

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats

# class RandomFeatureGaussianProcess(nn.Module):
#     def __init__(self,
#                  in_features: int,
#                  out_features: int,
#                  num_inducing: int = 1024,
#                  kernel_scale: float = 1,
#                  normalize_input: bool = False,
#                  scale_random_features: bool = False,
#                  cov_momentum: float = .999,
#                  ridge_penalty: float = 1e-6,
#                  mean_field_factor: float = 1,
#                  ):
#         super().__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_inducing = num_inducing
#
#         # scale inputs
#         self.kernel_scale = kernel_scale
#         self.normalize_input = normalize_input
#         self.input_scale = (1/math.sqrt(kernel_scale) if kernel_scale is not None else None)
#
#         # Random features
#         self.scale_random_features = scale_random_features
#         self.random_feature_scale = math.sqrt(2./float(num_inducing))
#
#         # Inference
#         self.mean_field_factor = mean_field_factor
#
#         # Covariance computation
#         self.ridge_penalty = ridge_penalty
#         self.cov_momentum = cov_momentum
#
#         # Define scale, weight and bias according to Eq. 7
#         # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
#         self.random_feature_linear = nn.Linear(in_features, num_inducing)
#         self.random_feature_linear.weight.requires_grad = False
#         self.random_feature_linear.bias.requires_grad = False
#         nn.init.normal_(self.random_feature_linear.weight, std=0.05)
#         nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)
#
#         # Define output layer according to Eq 8. For imagenet init with normal std=0.01?
#         self.beta = nn.Linear(num_inducing, out_features, bias=False)
#
#         self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
#         self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)
#
#     def forward(self, features, return_cov=False, update_precision=True):
#         if self.normalize_input:
#             features = self.layer_norm(features)
#         elif self.input_scale is not None:
#             features = features * self.input_scale
#
#         # Get gp features according to Eq. 7
#         phi = torch.cos(self.random_feature_linear(features))
#         # See: https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/wide_resnet_sngp.py#L207
#         if self.scale_random_features:
#             phi = self.random_feature_scale * phi
#
#         # Eq. 8
#         logits = self.beta(phi)
#
#         if update_precision:
#             self.update_precision_matrix(phi)
#
#         if return_cov:
#             cov = self.compute_predictive_covariance(phi)
#             return logits, cov
#         return logits
#
#     def reset_covariance(self):
#         device = self.precision_matrix.device
#         self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)
#         self.precision_matrix.to(device)
#
#     def update_precision_matrix(self, phi):
#         precision_matrix_minibatch = torch.matmul(phi.T, phi)
#         if self.cov_momentum > 0:
#             batch_size = len(phi)
#             precision_matrix_minibatch = precision_matrix_minibatch / batch_size
#             precision_matrix_new = (self.cov_momentum * self.precision_matrix +
#                                     (1-self.cov_momentum) * precision_matrix_minibatch)
#         else:
#             precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
#         self.precision_matrix = nn.Parameter(precision_matrix_new, requires_grad=False)
#
#     def compute_predictive_covariance(self, phi):
#         covariance_matrix_feature = torch.linalg.pinv(self.precision_matrix)
#         out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
#         covariance_matrix_gp = torch.matmul(phi, out)
#         return covariance_matrix_gp
