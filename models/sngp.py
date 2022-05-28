import math
import copy

import torch
import torch.nn as nn

from utils import MetricLogger, SmoothedValue
from metrics import ood, generalization


class SNGP(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 in_features: int,
                 num_inducing: int,
                 num_classes: int,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 auto_scale_kernel: bool = False,
                 momentum: float = .999,
                 ridge_penalty: float = 1e-6,
                 ):
        super().__init__()
        self.model = model
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        self.input_scale = (1/math.sqrt(kernel_scale) if kernel_scale is not None else None)
        self.feature_scale = math.sqrt(2./float(num_inducing))
        self.kernel_scale = kernel_scale

        self.normalize_input = normalize_input
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(in_features)

        # Define scale, weight and bias according to Eq. 7
        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        nn.init.normal_(self.random_feature_linear.weight)
        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

        # Scale the random features according to SNGP, bigger sigma equals bigger kernel
        if auto_scale_kernel:
            kernel_scale = math.sqrt(in_features / 2) if kernel_scale is None else kernel_scale
            self.random_feature_linear.weight.data /= self.kernel_scale

        # Define output layer according to Eq 8.
        self.beta = nn.Linear(num_inducing, num_classes, bias=False)
        nn.init.normal_(self.beta.weight, std=.01)  # std needs to be small here

        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)

    def reset_covariance(self):
        device = self.precision_matrix.device
        self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)
        self.precision_matrix.to(device)

    def update_precision_matrix(self, phi):
        precision_matrix_minibatch = torch.matmul(phi.T, phi)
        if self.momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.momentum * self.precision_matrix +
                                    (1-self.momentum) * precision_matrix_minibatch)
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
        elif self.input_scale is not None:
            features = features * self.input_scale

        # Get gp features according to Eq. 7
        phi = torch.cos(self.random_feature_linear(features))
        phi = self.feature_scale * phi

        # Eq. 8
        logits = self.beta(phi)

        if update_precision:
            self.update_precision_matrix(phi)

        if return_cov:
            cov = self.compute_predictive_covariance(phi)
            return logits, cov
        return logits

    @torch.no_grad()
    def forward_sample(self, x, n_draws=10):
        # Transform to feature space
        _, features = self.model(x, return_features=True)

        if self.normalize_input:
            features = self.layer_norm(features)

        features = features * self.input_scale
        phi = torch.cos(self.random_feature_linear(features))
        phi = self.feature_scale * phi

        logits_mean = self.beta(phi)
        logits_cov = self.compute_predictive_covariance(phi)

        n_samples, n_classes = logits_mean.shape
        logits_cov += torch.eye(n_samples)*1e-3
        logits_sampled = torch.empty((n_draws, n_samples, n_classes))
        for k in range(n_classes):
            dist = torch.distributions.MultivariateNormal(loc=logits_mean[:, k], covariance_matrix=logits_cov)
            logits_sampled[:, :, k] = dist.sample((n_draws,))

        return logits_sampled


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.reset_covariance()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    # Train the epoch
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
        logits, cov = model(inputs, return_cov=True, update_precision=False)
        logits_id.append(mean_field_logits(logits, cov))
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
        logits, cov = model(inputs, return_cov=True, update_precision=False)
        logits_ood.append(mean_field_logits(logits, cov))
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
