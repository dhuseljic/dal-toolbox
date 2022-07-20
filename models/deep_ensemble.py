import torch
import torch.nn as nn

from utils import MetricLogger
from metrics import metrics, generalization

class EnsembleLR(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, lr_schedulers):
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


class EnsembleOptimizer(torch.optim.SGD):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def state_dict(self) -> dict:
        return [optim.state_dict() for optim in self.optimizers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for optim, state_dict in zip(self.optimizers, state_dict_list):
            optim.load_state_dict(state_dict)

    def __iter__(self):
        for optim in self.optimizers:
            yield optim


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.mean(self.forward_sample(x), dim=0)

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    train_stats = {}
    model.train()
    model.to(device)

    for i_member, (member, optim) in enumerate(zip(model.models, optimizer)):
        
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
        train_stats.update({f"train_model{i_member}_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()})
    return train_stats



@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    model.eval()
    model.to(device)
    test_stats = {}

    # Get in-domain logits and targets for test set
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Get out-of-domain logits and targets for test set
    logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model(inputs))
    logits_ood = torch.cat(logits_ood, dim=0).cpu()

    test_stats = metrics.get_test_stats(logits_id, targets_id, logits_ood)
    
    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats