import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger
from metrics import generalization, calibration, ood


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        raise ValueError('Forward method should only be used on ensemble members.')

    def forward_sample(self, x):
        logits = []
        for m in self.models:
            logits.append(m(x))
        return torch.stack(logits)

    def __iter__(self):
        for m in self.models:
            yield m

    def __len__(self):
        return len(self.models)


class EnsembleLRScheduler:
    def __init__(self, lr_schedulers: list):
        self.lr_schedulers = lr_schedulers

    def step(self):
        for lrs in self.lr_schedulers:
            lrs.step()

    def state_dict(self) -> dict:
        return [lrs.state_dict() for lrs in self.lr_schedulers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for lrs, state_dict in zip(self.lr_schedulers, state_dict_list):
            lrs.load_state_dict(state_dict)

    def __iter__(self):
        for lrs in self.lr_schedulers:
            yield lrs


class EnsembleOptimizer:
    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    def state_dict(self) -> dict:
        return [optim.state_dict() for optim in self.optimizers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for optim, state_dict in zip(self.optimizers, state_dict_list):
            optim.load_state_dict(state_dict)

    def __iter__(self):
        for optim in self.optimizers:
            yield optim


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    train_stats = {}
    model.train()
    model.to(device)

    for i_member, (member, optim) in enumerate(zip(model, optimizer)):
        metric_logger = MetricLogger(delimiter=" ")
        header = f"Epoch [{epoch}] Model [{i_member}] " if epoch is not None else f"Model [{i_member}] "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = member(inputs)

            loss = criterion(outputs, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        train_stats.update({f"train_{k}_model{i_member}": meter.global_avg for k, meter, in metric_logger.meters.items()})
    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Get logits and targets for in-domain-test-set (Number of Members x Number of Samples x Number of Classes)
    ensemble_logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        ensemble_logits_id.append(model.forward_sample(inputs))
        targets_id.append(targets)
    
    # Transform to tensor
    ensemble_logits_id = torch.cat(ensemble_logits_id, dim=1).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Transform into probabilitys 
    ensemble_probas_id = ensemble_logits_id.softmax(dim=-1)

    # Average of probas per sample
    mean_probas_id = torch.mean(ensemble_probas_id, dim=0)

    # Confidence- and entropy-Scores of in domain set logits
    conf_id, _ = mean_probas_id.max(-1)
    entropy_id = ood.entropy_fn(mean_probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(torch.log(mean_probas_id), targets_id, (1,))[0].item()
    loss = criterion(torch.log(mean_probas_id), targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()
    ensemble_cross_entropy = calibration.EnsembleCrossEntropy()(ensemble_logits_id, targets_id).item()
    gibbs_cross_entropy = calibration.GibsCrossEntropy()(ensemble_logits_id, targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

    metrics = {
        "acc1": acc1,
        "loss": loss,
        "nll": nll,
        "ensemble_cross_entropy": ensemble_cross_entropy,
        "gibbs_cross_entropy": gibbs_cross_entropy,
        "tce": tce,
        "mce": mce
    }

    for name, dataloader_ood in dataloaders_ood.items():
            # Repeat for out-of-domain-test-set
        ensemble_logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            ensemble_logits_ood.append(model.forward_sample(inputs))
        ensemble_logits_ood = torch.cat(ensemble_logits_ood, dim=1).cpu()
        ensemble_probas_ood = ensemble_logits_ood.softmax(dim=-1)
        mean_probas_ood = torch.mean(ensemble_probas_ood, dim=0)

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