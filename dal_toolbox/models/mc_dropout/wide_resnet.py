import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from ..utils.mcdropout import MCDropoutModule, ConsistentMCDropout2d
from ...metrics import generalization, ood, calibration
from ...utils import MetricLogger, SmoothedValue


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class DropoutWideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(DropoutWideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = ConsistentMCDropout2d(dropout_rate)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def dropout_wide_resnet_28_10(num_classes, n_mc_passes, dropout_rate=0.3):
    model = DropoutWideResNet(
        depth=28,
        widen_factor=10,
        k=n_mc_passes,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )
    return model


class DropoutWideResNet(MCDropoutModule):
    def __init__(self, depth, widen_factor, num_classes=10, k=None, dropout_rate=0.1):
        super(MCDropoutModule, self).__init__()
        self.in_planes = 16
        self.depth = depth
        self.widen_factor = widen_factor
        self.k = k

        assert ((self.depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (self.depth-4)/6
        w = self.widen_factor

        nStages = [16, 16*w, 32*w, 64*w]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(DropoutWideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(DropoutWideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(DropoutWideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(out)
        return out

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return self.forward(mc_input_BK)


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
    dropout_logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        dropout_logits_id.append(model.mc_forward(inputs, model.k))
        targets_id.append(targets)

    # Transform to tensor
    dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Transform into probabilitys
    dropout_probas_id = dropout_logits_id.softmax(dim=-1)

    # Average of probas per sample
    mean_probas_id = torch.mean(dropout_probas_id, dim=1)

    # Confidence- and entropy-Scores of in domain set logits
    conf_id, _ = mean_probas_id.max(-1)
    entropy_id = ood.entropy_fn(mean_probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(torch.log(mean_probas_id), targets_id, (1,))[0].item()
    prec = generalization.avg_precision(mean_probas_id, targets_id)
    loss = criterion(torch.log(mean_probas_id), targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

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
        dropout_logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            dropout_logits_ood.append(model.mc_forward(inputs, model.k))
        dropout_logits_ood = torch.cat(dropout_logits_ood, dim=0).cpu()
        dropout_probas_ood = dropout_logits_ood.softmax(dim=-1)
        mean_probas_ood = torch.mean(dropout_probas_ood, dim=1)

        # Confidence- and entropy-Scores of out of domain logits
        conf_ood, _ = mean_probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(mean_probas_ood)

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Training: Epoch {epoch}"
    model.to(device)
    model.train()

    for X_batch, y_batch in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        loss = criterion(out, y_batch)
        batch_size = X_batch.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, = generalization.accuracy(out.softmax(dim=-1), y_batch, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats
