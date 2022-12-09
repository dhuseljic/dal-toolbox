import torch
import torch.nn as nn

from ..metrics import generalization, ood, calibration
from ..utils import MetricLogger, SmoothedValue


class LeNet(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        super(LeNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
        )
        self.last = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.linear_block(x)
        features = x.detach()
        out = self.last(x)
        if return_features:
            out = (out, features)
        return out

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        return probas

    @torch.inference_mode()
    def get_representation(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for samples, _ in dataloader:
            _, features = self(samples.to(device), return_features=True)
            all_features.append(features.cpu())
        features = torch.cat(all_features)
        return features


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
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


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
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
            logits_ood.append(model(inputs))
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
