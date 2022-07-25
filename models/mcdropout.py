import torch
import torch.nn as nn
from metrics import generalization, calibration, ood
from utils import MetricLogger, SmoothedValue


class MCDropout(nn.Module):
    def __init__(self, model, k):
        super().__init__()
        self.model = model
        self.k = k

@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id_k, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id_k.append(model.model.mc_forward(inputs, model.k))
        targets_id.append(targets)
    logits_id_k = torch.cat(logits_id_k, dim=0).cpu()
    logits_id = torch.mean(logits_id_k, dim=1)
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Forward prop out of distribution
    logits_ood_k = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood_k.append(model.model.mc_forward(inputs, model.k))
    logits_ood_k = torch.cat(logits_ood_k, dim=0).cpu()
    logits_ood = torch.mean(logits_ood_k, dim=1) # Mean over dim=1 since Bayesian Dropout Module stores the k passes in second dimension

    # Test Loss and Accuracy for in domain testset
    acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
    loss = criterion(logits_id, targets_id).item()

    # Confidence- and entropy-Scores of in domain and out of domain logits
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    # Area under the Precision-Recall-Curve
    entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
    conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

    # Area under the Receiver-Operator-Characteristic-Curve
    entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
    conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()
    
    return {f"test_{k}": v for k, v in {
        "acc1":acc1,
        "loss":loss,
        "nll":nll,
        "entropy_auroc":entropy_auroc,
        "entropy_aupr":entropy_aupr,
        "conf_auroc":conf_auroc,
        "conf_aupr":conf_aupr,
        "tce":tce,
        "mce":mce
    }.items()}


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Training: Epoch {epoch}"
    model.to(device)
    model.train()

    #TODO: Train Model with or without Dropout? What are optimal ResNet Params for Training with 2dropout for each dataset?

    for X_batch, y_batch in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model.model(X_batch)
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
