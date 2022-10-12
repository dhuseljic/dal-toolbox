import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import generalization, calibration, ood
from .utils.spectral_norm import SpectralConv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, self.expansion*planes, kernel_size=1, spectral_norm=spectral_norm, norm_bound=norm_bound,
                               n_power_iterations=n_power_iterations, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def resnet18_sngp(num_classes,
                  norm_bound,
                  n_power_iterations=1,
                  spectral_norm=True,
                  num_inducing=1024,
                  kernel_scale=1,
                  normalize_input=False,
                  scale_random_features=True,
                  mean_field_factor=math.pi/8,
                  cov_momentum=-1,
                  ridge_penalty=1,
                  ):
    resnet18_blocks = [2, 2, 2, 2]
    return ResNetSNGP(
        num_classes=num_classes,
        block=BasicBlock,
        num_blocks=resnet18_blocks,
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_inducing=num_inducing,
        kernel_scale=kernel_scale,
        normalize_input=normalize_input,
        scale_random_features=scale_random_features,
        mean_field_factor=mean_field_factor,
        cov_momentum=cov_momentum,
        ridge_penalty=ridge_penalty
    )


class ResNetSNGP(nn.Module):
    def __init__(self,
                 num_classes,
                 block,
                 num_blocks,
                 spectral_norm=True,
                 norm_bound=1,
                 n_power_iterations=1,
                 num_inducing=1024,
                 kernel_scale=1,
                 normalize_input=False,
                 scale_random_features=True,
                 mean_field_factor=math.pi/8,
                 cov_momentum=-1,
                 ridge_penalty=1,
                 ):
        super(ResNetSNGP, self).__init__()
        self.in_planes = 64
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        self.spectral_norm = spectral_norm

        # Init layer does not have a kernel size of 7 since cifar has a smaller
        # size of 32x32
        self.conv1 = SpectralConv2d(3, 64, kernel_size=3, spectral_norm=self.spectral_norm, norm_bound=self.norm_bound,
                                    n_power_iterations=self.n_power_iterations, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Last Layer will be a random feature GP
        self.output_layer = RandomFeatureGaussianProcess(
            in_features=512*block.expansion,
            out_features=num_classes,
            num_inducing=num_inducing,
            kernel_scale=kernel_scale,
            normalize_input=normalize_input,
            scale_random_features=scale_random_features,
            mean_field_factor=mean_field_factor,
            cov_momentum=cov_momentum,
            ridge_penalty=ridge_penalty,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, self.spectral_norm, self.norm_bound, self.n_power_iterations)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def reset_precision_matrix(self):
        self.output_layer.reset_precision_matrix()

    def forward(self, x, mean_field=False, return_cov=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.features = out
        if mean_field:
            out = self.output_layer.forward_mean_field(self.features)
        else:
            out = self.output_layer(self.features, return_cov=return_cov)
        return out


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_inducing: int = 1024,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 scale_random_features: bool = True,
                 mean_field_factor: float = math.pi/8,
                 cov_momentum: float = -1,
                 ridge_penalty: float = 1,
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing

        # scale inputs
        self.kernel_scale = kernel_scale
        self.normalize_input = normalize_input
        self.layer_norm = nn.LayerNorm(in_features)

        # Random features
        self.scale_random_features = scale_random_features

        # Inference
        self.mean_field_factor = mean_field_factor

        # Covariance computation
        self.ridge_penalty = ridge_penalty
        self.cov_momentum = cov_momentum

        self.random_features = RandomFourierFeatures(
            in_features=self.in_features,
            num_inducing=self.num_inducing,
            kernel_scale=self.kernel_scale,
            scale_features=self.scale_random_features,
        )
        self.random_features.reset_parameters(std_init=0.05)

        # Define output layer according to Eq 8. For imagenet init with normal std=0.01?
        self.beta = nn.Linear(num_inducing, out_features, bias=False)

        # precision matrix
        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.cov_mat = None

    def forward(self, features, return_cov=False):
        if self.normalize_input:
            features = self.layer_norm(features)

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
        multiplier = probas_max * (1-probas_max)
        precision_matrix_minibatch = torch.matmul(
            multiplier*phi.T, phi
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
        logits_scaled = model(inputs, mean_field=True)
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
        "acc1": acc1,
        "loss": loss,
        "nll": nll,
        "tce": tce,
        "mce": mce
    }

    for name, dataloader_ood in dataloaders_ood.items():
        # Forward prop out of distribution
        logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_scaled = model(inputs, mean_field=True)
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