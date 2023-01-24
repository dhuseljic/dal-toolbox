import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from ...utils import MetricLogger, SmoothedValue
from ...metrics import generalization, calibration, ood
from ..utils.spectral_normalization import SpectralConv2d
from ..utils.random_features import RandomFeatureGaussianProcess


def conv3x3(in_planes, out_planes, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
    return SpectralConv2d(in_planes, out_planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                          n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = SpectralConv2d(in_planes, planes, kernel_size=3, spectral_norm=spectral_norm,
                                    norm_bound=norm_bound, n_power_iterations=n_power_iterations, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, kernel_size=3, spectral_norm=spectral_norm,
                                    norm_bound=norm_bound, n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, planes, kernel_size=1, spectral_norm=spectral_norm,
                               norm_bound=norm_bound, n_power_iterations=n_power_iterations, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNetSNGP(nn.Module):
    def __init__(self,
                 depth,
                 widen_factor,
                 dropout_rate,
                 num_classes,
                 input_shape,
                 norm_bound,
                 n_power_iterations,
                 spectral_norm=True,
                 num_inducing=1024,
                 kernel_scale=1,
                 normalize_input=False,
                 scale_random_features=True,
                 random_feature_type='orf',
                 mean_field_factor=math.pi/8,
                 cov_momentum=-1,
                 ridge_penalty=1
                 ):
        super(WideResNetSNGP, self).__init__()
        self.in_planes = 16
        self.depth = depth
        self.widen_factor = widen_factor
        self.input_shape = input_shape

        # Spectral norm params
        self.spectral_norm = spectral_norm
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations

        assert ((self.depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (self.depth-4)/6
        k = self.widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0], spectral_norm=self.spectral_norm,
                             norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        # Last Layer will be a random feature GP
        self.output_layer = RandomFeatureGaussianProcess(
            in_features=nStages[3],
            out_features=num_classes,
            num_inducing=num_inducing,
            kernel_scale=kernel_scale,
            normalize_input=normalize_input,
            random_feature_type=random_feature_type,
            scale_random_features=scale_random_features,
            mean_field_factor=mean_field_factor,
            cov_momentum=cov_momentum,
            ridge_penalty=ridge_penalty,
        )
        self.init_spectral_norm(input_shape=self.input_shape)

    @torch.no_grad()
    def init_spectral_norm(self, input_shape):
        # Currently needed for conv layers
        dummy_input = torch.randn((1, *input_shape))
        self(dummy_input)

    def reset_precision_matrix(self):
        self.output_layer.reset_precision_matrix()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, dropout_rate=dropout_rate, stride=stride,
                          spectral_norm=self.spectral_norm, norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, mean_field=False, return_cov=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if mean_field:
            out = self.output_layer.forward_mean_field(out)
        else:
            out = self.output_layer(out, return_cov=return_cov)
        return out


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
    prec = generalization.avg_precision(probas_id, targets_id)
    loss = criterion(logits_id, targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

    metrics = {
        "acc1": acc1,
        "prec": prec,
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
