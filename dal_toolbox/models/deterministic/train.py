import torch
import torch.nn as nn
import torch.nn.functional as F

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
# def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
#     model.train()
#     model.to(device)
#
#     metric_logger = MetricLogger(delimiter=" ")
#     metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
#     header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "
#
#     # Train the epoch
#     for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
#         inputs, targets = inputs.to(device), targets.to(device)
#
#         outputs = model(inputs)
#
#         loss = criterion(outputs, targets)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         batch_size = inputs.shape[0]
#         acc1, = generalization.accuracy(outputs, targets, topk=(1,))
#         metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
#         metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
#     train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
#
#     return train_stats


# SSL training methods
def train_one_epoch_pseudolabel(model, labeled_loader, unlabeled_loader, criterion, optimizer, lr_scheduler, n_iter, p_cutoff, lambda_u, device,
                                use_hard_labels=True, unsup_warmup=.4, epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    unlabeled_iter = iter(unlabeled_loader)

    i_iter = epoch*len(labeled_loader)

    for (x_lb, y_lb) in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)

        # Supervised
        logits_lb = model(x_lb)
        sup_loss = criterion(logits_lb, y_lb)

        # Unsupervised
        x_ulb, _ = next(unlabeled_iter)
        x_ulb = x_ulb.to(device)

        # Outputs of unlabeled data but without batch norm
        bn_backup = freeze_bn(model)
        logits_ulb = model(x_ulb)
        unfreeze_bn(model, bn_backup)

        # Generate pseudo labels and mask
        probas_ulb = torch.softmax(logits_ulb.detach(), dim=-1)
        max_probas, pseudo_label = torch.max(probas_ulb, dim=-1)
        mask = max_probas.ge(p_cutoff)

        if not use_hard_labels:
            T = 1
            probas = torch.softmax(logits_ulb.detach()/T, dim=-1)
            pseudo_label, _ = probas.max(-1)

        unsup_loss = torch.mean(F.cross_entropy(logits_ulb, pseudo_label, reduction='none') * mask)
        unsup_warmup_factor = np.clip(i_iter / (unsup_warmup*n_iter), a_min=0, a_max=1)
        i_iter += 1

        # Loss thats used for backpropagation
        loss = sup_loss + unsup_warmup_factor * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             mask_ratio=mask.float().mean().item(), unsup_warmup_factor=unsup_warmup_factor, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


def train_one_epoch_pimodel(model, labeled_loader, unlabeled_loader_weak_1, unlabeled_loader_weak_2, criterion, optimizer, lr_scheduler, n_iter, lambda_u, device, unsup_warmup=.4,
                            epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    unlabeled_iter1 = iter(unlabeled_loader_weak_1)
    unlabeled_iter2 = iter(unlabeled_loader_weak_2)

    i_iter = epoch*len(labeled_loader)
    for x_lb, y_lb in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)

        # Supervised
        logits_lb = model(x_lb)
        sup_loss = criterion(logits_lb, y_lb)

        # Unsupervised Loss
        x_ulb_weak_1, _ = next(unlabeled_iter1)
        x_ulb_weak_1 = x_ulb_weak_1.to(device)
        x_ulb_weak_2, _ = next(unlabeled_iter2)
        x_ulb_weak_2 = x_ulb_weak_2.to(device)

        # Outputs of unlabeled data without batch norm
        bn_backup = freeze_bn(model)
        logits_ulb_weak_1 = model(x_ulb_weak_1)
        logits_ulb_weak_2 = model(x_ulb_weak_2)
        unfreeze_bn(model, bn_backup)

        unsup_loss = F.mse_loss(logits_ulb_weak_2.softmax(-1), logits_ulb_weak_1.detach().softmax(-1))
        unsup_warmup_factor = np.clip(i_iter / (unsup_warmup*n_iter), a_min=0, a_max=1)
        i_iter += 1

        # Loss thats used for backpropagation
        loss = sup_loss + unsup_warmup_factor * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             unsup_warmup_factor=unsup_warmup_factor, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


def train_one_epoch_fixmatch(model, optimizer, lr_scheduler, criterion, device, epoch, labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, 
                    p_cutoff, lambda_u, T=1):
    model.to(device)
    model.train()

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    iterator_unlabeled_weak_aug = iter(unlabeled_loader_weak)
    iterator_unlabeled_strong_aug = iter(unlabeled_loader_strong)

    for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=50, header=header):
        (x_w, y_w), (x_s, _)  = next(iterator_unlabeled_weak_aug), next(iterator_unlabeled_strong_aug)
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_w = x_w.to(device)
        x_s = x_s.to(device)
        y_w = y_w.to(device)

        # Get all necesseracy model outputs
        logits_l = model(x_l)
        with torch.no_grad():
            logits_w = model(x_w)
        logits_s = model(x_s)

        # Calculate pseudolabels and mask
        probas_w = (logits_w/T).softmax(-1)
        y_probs, y_ps = probas_w.max(-1)
        mask = y_probs.ge(p_cutoff).to(device)

        # Calculate Loss
        loss_l = criterion(logits_l, y_l)
        loss_u = criterion(logits_s, y_ps) * mask
        loss = loss_l.mean() + lambda_u * loss_u.mean()

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Collect Metrics
        batch_size, ulb_batch_size = x_l.shape[0], x_s.shape[0]
        acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
        pseudo_acc1, = generalization.accuracy(logits_w, y_w, topk=(1,))

        # Log Metrics
        metric_logger.update(loss=loss.item(), supervised_loss=loss_l.mean().item(), lr=optimizer.param_groups[0]["lr"],
                             unsupervised_loss=loss_u.mean().item(), mask_ratio=mask.float().mean().item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=ulb_batch_size)
        
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


def train_one_epoch_flexmatch(model, optimizer, lr_scheduler, criterion, device, epoch, labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, 
                    unlabeled_loader_indices, p_cutoff, lambda_u, fmth, T=1):
    model.to(device)
    model.train()

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    iterator_unlabeled_weak_aug = iter(unlabeled_loader_weak)
    iterator_unlabeled_strong_aug = iter(unlabeled_loader_strong)
    iterator_unlabeled_indices = iter(unlabeled_loader_indices)

    for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=50, header=header):
        (x_w, y_w), (x_s, _), idx  = next(iterator_unlabeled_weak_aug), next(iterator_unlabeled_strong_aug), next(iterator_unlabeled_indices)
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_w = x_w.to(device)
        x_s = x_s.to(device)
        y_w = y_w.to(device)
        idx = idx.to(device)    

        # Get all necesseracy model outputs
        logits_l = model(x_l)
        with torch.no_grad():
            logits_w = model(x_w)
        logits_s = model(x_s)

        # Calculate pseudolabels and mask
        probas_w = (logits_w/T).softmax(-1)
        _, y_ps = probas_w.max(-1)
        mask = fmth.masking(p_cutoff, probas_w, idx)

        # Calculate Loss
        loss_l = criterion(logits_l, y_l)
        loss_u = criterion(logits_s, y_ps) * mask
        loss = loss_l.mean() + lambda_u * loss_u.mean()

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Collect Metrics
        batch_size, ulb_batch_size = x_l.shape[0], x_s.shape[0]
        acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
        pseudo_acc1, = generalization.accuracy(logits_w, y_w, topk=(1,))

        # Log Metrics
        metric_logger.update(loss=loss.item(), supervised_loss=loss_l.mean().item(), lr=optimizer.param_groups[0]["lr"],
                             unsupervised_loss=loss_u.mean().item(), mask_ratio=mask.float().mean().item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=ulb_batch_size)
        
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats
