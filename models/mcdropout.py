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

    # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
    dropout_logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        dropout_logits_id.append(model.model.mc_forward(inputs, model.k))
        targets_id.append(targets)
    
    # Transform to tensor
    dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Transform into probabilitys 
    dropout_probas_id = dropout_logits_id.softmax(dim=-1)

    # Average of probas per sample
    mean_probas_id = torch.mean(dropout_probas_id, dim=1)

    # Repeat for out-of-domain-test-set
    dropout_logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        dropout_logits_ood.append(model.model.mc_forward(inputs, model.k))
    dropout_logits_ood = torch.cat(dropout_logits_ood, dim=0).cpu()
    dropout_probas_ood = dropout_logits_ood.softmax(dim=-1)
    mean_probas_ood = torch.mean(dropout_probas_ood, dim=1)


    # Test Loss and Accuracy for in domain testset
    acc1 = generalization.accuracy(mean_probas_id, targets_id, (1,))[0].item()
    loss = criterion(torch.log(mean_probas_id), targets_id).item()

    # Confidence- and entropy-Scores of in domain and out of domain logits
    conf_id, _ = mean_probas_id.max(-1)
    conf_ood, _ = mean_probas_ood.max(-1)
    entropy_id = ood.entropy_fn(mean_probas_id)
    entropy_ood = ood.entropy_fn(mean_probas_ood)

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()

    # Area under the Precision-Recall-Curve
    entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
    conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

    # Area under the Receiver-Operator-Characteristic-Curve
    entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
    conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()
    
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
