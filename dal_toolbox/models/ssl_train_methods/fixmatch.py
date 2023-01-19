import torch
import torch.nn.functional as F

from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue

def train_one_epoch(model, dataloaders, criterion, optimizer, device, use_hard_labels,
                    lambda_u, p_cutoff, use_cat, T):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    for (x_lb, y_lb), (x_ulb_weak, _), (x_ulb_strong, _) in zip(dataloaders['train_sup'], dataloaders['train_unsup_weak'], dataloaders['train_unsup_strong']):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)
        x_ulb_weak = x_ulb_weak.to(device)
        x_ulb_strong = x_ulb_strong.to(device)

        # Forward Propagation of all samples
        if use_cat:
            num_lb = x_lb.shape[0]
            outputs = model(torch.cat((x_lb, x_ulb_weak, x_ulb_strong)))
            logits_lb = outputs[:num_lb]
            logits_ulb_weak, logits_ulb_strong = outputs[num_lb:].chunk(2)
        else:
            logits_lb = model(x_lb) 
            logits_ulb_strong = model(x_ulb_strong)
            with torch.no_grad():
               logits_ulb_weak = model(x_ulb_weak)

        # Generate pseudo labels and mask
        if use_hard_labels:
            probas_ulb_weak = torch.softmax(logits_ulb_weak.detach(), dim=-1)
        else:
            T = 1.0
            probas_ulb_weak = torch.softmax(logits_ulb_weak.detach()/T, dim=-1)
        max_probas_weak, pseudo_labels = torch.max(probas_ulb_weak, dim=-1)
        mask = max_probas_weak.ge(p_cutoff)

        # Loss
        sup_loss = criterion(logits_lb, y_lb)
        unsup_loss = (F.cross_entropy(logits_ulb_strong, pseudo_labels, reduction='none') * mask).mean()
        total_loss = sup_loss + lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(
            sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(), 
            total_loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"]
            )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats