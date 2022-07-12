import torch
import torch.nn as nn
from metrics import ood, generalization
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
    test_stats = {}

    # Forward prop in distribution
    logits_id_k, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id_k.append(model.model.mc_forward(inputs, model.k))
        targets_id.append(targets)
    logits_id_k = torch.cat(logits_id_k, dim=0).cpu()
    print(logits_id_k.shape)
    logits_id = torch.mean(logits_id_k, dim=1)
    targets_id = torch.cat(targets_id, dim=0).cpu()
    print(logits_id.shape)


    # Update test stats
    loss = criterion(logits_id, targets_id)
    acc1, = generalization.accuracy(logits_id.softmax(dim=-1), targets_id, (1,))
    test_stats.update({'loss': loss.item()})
    test_stats.update({'acc1': acc1.item()})

    # Forward prop out of distribution
    logits_ood_k = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood_k.append(model.model.mc_forward(inputs, model.k))
    logits_ood_k = torch.cat(logits_ood_k, dim=0).cpu()
    # Mean over dim=1 since Bayesian Dropout Module stores the k passes in second dimension
    logits_ood = torch.mean(logits_ood_k, dim=1)

    # Update test stats
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    test_stats.update({'auroc': ood.ood_auroc(entropy_id, entropy_ood)})

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood.ood_auroc(1-conf_id, 1-conf_ood)})

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats


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
