import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import ood, generalization, calibration


class SNGP2(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 in_features: int,
                 num_classes: int,
                 num_inducing: int = 1024,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 scale_random_features: bool = False,
                 mean_field_factor: float = math.pi/8,
                 cov_momentum: float = -1,
                 ridge_penalty: float = 1,
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
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.cov_mat = None

        self.sampled_betas = None

    @property
    def covariance_matrix(self):
        device = self.precision_matrix.data.device
        if self.cov_mat is None:
            u = torch.linalg.cholesky(self.precision_matrix.data)
            self.cov_mat = torch.cholesky_inverse(u)
        return self.cov_mat.to(device)

    def reset_precision_matrix(self):
        device = self.precision_matrix.device
        self.precision_matrix.data = copy.deepcopy(self.init_precision_matrix)
        self.precision_matrix.to(device)
        self.cov_mat = None

    @torch.no_grad()
    def update_precision_matrix(self, phi, logits):
        probas = logits.softmax(-1)
        probas_max = probas.max(1)[0]
        precision_matrix_minibatch = torch.matmul(
            probas_max * (1-probas_max) * phi.T, phi
        )
        if self.cov_momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix.data +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix.data + precision_matrix_minibatch
        self.precision_matrix.data = precision_matrix_new
        self.cov_mat = None

    def compute_predictive_covariance(self, phi):
        covariance_matrix_feature = self.covariance_matrix
        out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(phi, out)
        return covariance_matrix_gp

    def forward(self, x, return_cov=False):
        _, features = self.model(x, return_features=True)

        if self.normalize_input:
            features = self.layer_norm(features)

        # Get gp features according to Eq. 7
        phi = self.random_features(features)

        # Eq. 8
        logits = self.beta(phi)

        if self.training:
            self.update_precision_matrix(phi, logits)
        if return_cov:
            cov = self.compute_predictive_covariance(phi)
            return logits, cov
        return logits

    @torch.no_grad()
    def forward_mean_field(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits

    def sample_betas(self, n_draws):
        dist = torch.distributions.MultivariateNormal(
            loc=self.beta.weight,
            precision_matrix=self.precision_matrix
        )
        self.sampled_betas = dist.sample(sample_shape=(n_draws,))

    @torch.no_grad()
    def forward_sample(self, x):
        _, features = self.model(x, return_features=True)
        if self.normalize_input:
            features = self.layer_norm(features)
        phi = self.random_features(features)
        logits_sampled = torch.einsum('nd,ekd->enk', phi, self.sampled_betas)
        return logits_sampled

    def forward_dirichlet(self, x, use_variance_correction=False):
        if self.training:
            raise ValueError("Call eval mode before!")
        # Get logit mean and covariance predictions.
        logits, cov = self(x, return_cov=True)
        var = torch.diag(cov)
        var = torch.clamp(var, min=1.e-5)

        # Zero mean correction.
        logits -= ((var * logits.sum(-1)) / (var * self.num_classes))[:, None]
        var *= (self.num_classes - 1) / self.num_classes

        # Optional variance correction.
        if use_variance_correction:
            c = var / math.sqrt(self.num_classes/2)
            logits /= c.sqrt()[:, None]
            var /= c

        # Compute alphas.
        sum_exp = torch.exp(-logits).sum(dim=1).unsqueeze(-1)
        alphas = (1 - 2/self.num_classes + logits.exp()/self.num_classes**2 * sum_exp) / var[:, None]
        return alphas


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

    def reset_parameters(self, std_init=1):
        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
        nn.init.normal_(self.random_feature_linear.weight, std=std_init)
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


def mean_field_logits(logits, cov, lmb=math.pi / 8):
    """Scale logits using the mean field approximation proposed by https://arxiv.org/abs/2006.07584"""
    if lmb is None or lmb < 0:
        return logits
    variances = torch.diag(cov).view(-1, 1) if cov is not None else 1
    logits_adjusted = logits / torch.sqrt(1 + lmb*variances)
    return logits_adjusted


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.reset_precision_matrix()
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


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
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

    # Confidence- and entropy-Scores of in domain set logits
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    entropy_id = ood.entropy_fn(probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
    loss = criterion(logits_id, targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

    metrics = {
        "acc1":acc1,
        "loss":loss,
        "nll":nll,
        "tce":tce,
        "mce":mce
    }

    for name, dataloader_ood in dataloaders_ood.items():
        # Forward prop out of distribution
        logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_scaled = model.forward_mean_field(inputs)
            logits_ood.append(logits_scaled)
        logits_ood = torch.cat(logits_ood, dim=0).cpu()

        # Confidence- and entropy-Scores of out of domain logits
        probas_ood = logits_ood.softmax(-1)
        conf_ood, _ = probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(probas_ood)
        
        # Area under the Precision-Recall-Curve
        entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
        conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

        # Area under the Receiver-Operator-Characteristic-Curve
        entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
        conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

        # Add to metrics
        metrics[name+"_entropy_auroc"] = entropy_auroc
        metrics[name+"_conf_auroc"] = conf_auroc
        metrics[name+"_entropy_aupr"] = entropy_aupr
        metrics[name+"_conf_aupr"] = conf_aupr

    return {f"test_{k}": v for k, v in metrics.items()}


@torch.no_grad()
def reweight(model, dataloader, device, lmb=1):
    model.eval()
    model.to(device)

    # Get all features and targets
    all_phis, all_targets = [], []
    for inputs, targets in dataloader:
        _, features = model.model(inputs.to(device), return_features=True)
        if model.normalize_input:
            features = model.layer_norm(features)
        phi = model.random_features(features)
        all_phis.append(phi.cpu())
        all_targets.append(targets)
    phis = torch.cat(all_phis)
    targets = torch.cat(all_targets)

    # Reweight
    model.cpu()
    mean = model.beta.weight.data.clone()
    cov = model.covariance_matrix.data.clone()
    targets_onehot = F.one_hot(targets, num_classes=model.num_classes)

    for phi, target_onehot in zip(phis, targets_onehot):
        for _ in range(lmb):
            tmp_1 = cov @ phi
            tmp_2 = torch.outer(tmp_1, tmp_1)

            # Compute new prediction.
            var = F.linear(phi, tmp_1)
            logits = F.linear(phi, mean)
            probas = logits.softmax(-1)
            probas_max = probas.max()

            # Update covariance matrix.
            num = probas_max * (1-probas_max)
            denom = 1 + num * var
            factor = num / denom
            cov_update = factor * tmp_2
            cov -= cov_update

            # Update mean.
            tmp_3 = F.linear(cov, phi)
            tmp_4 = (target_onehot - probas)
            mean += torch.outer(tmp_4, tmp_3)

            # Undo cov update.
            cov += cov_update

            # Compute new prediction.
            var = F.linear(phi, tmp_1)
            logits = F.linear(phi, mean)
            probas = logits.softmax(-1)
            probas_max = probas.max()

            # Update covariance matrix.
            num = probas_max * (1 - probas_max)
            denom = 1 + num * var
            factor = num / denom
            cov_update = factor * tmp_2
            cov -= cov_update

    model_reweighted = copy.deepcopy(model)
    model_reweighted.beta.weight.data = mean
    model_reweighted.covariance_matrix.data = cov
    return model_reweighted
