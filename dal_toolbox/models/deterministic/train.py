import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from ..utils import unfreeze_bn, freeze_bn
from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

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

    metric_logger.synchronize_between_processes()
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


def train_one_epoch_bertmodel(model, dataloader, epoch, optimizer, scheduler, criterion, device, print_freq=25):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        batch = batch.to(device)
        targets = batch['labels']

        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_size = targets.size(0)
        batch_acc, = generalization.accuracy(logits, targets, topk=(1,))

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)

    # save global (epoch) stats: take average of all the saved batch
    train_stats = {f"train_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Train Loss: {train_stats['train_loss_epoch']:.4f}, \
        Train Accuracy: {train_stats['train_batch_acc_epoch']:.4f}")
    print("--"*40)
    return train_stats