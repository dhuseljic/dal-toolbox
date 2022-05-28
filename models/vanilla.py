import torch
# import torch.nn as nn

from metrics import ood, generalization
from utils import MetricLogger, SmoothedValue

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
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    model.eval()
    model.to(device)
    test_stats = {}

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Update test stats
    loss = criterion(logits_id, targets_id)
    acc1, = generalization.accuracy(logits_id, targets_id, (1,))
    test_stats.update({'loss': loss.item()})
    test_stats.update({'acc1': acc1.item()})

    # Forward prop out of distribution
    logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model(inputs))
    logits_ood = torch.cat(logits_ood, dim=0).cpu()

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