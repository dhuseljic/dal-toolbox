import torch
import torch.nn.functional as F

from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue
from ..utils.pseudo_labels import generate_mask, generate_pseudo_labels, freeze_bn, unfreeze_bn


def train_one_epoch(model, dataloaders, criterion, optimizer, device, n_epochs, unsup_warmup, 
                    lambda_u, p_cutoff, use_hard_labels, epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    for (x_lb, y_lb), (x_ulb, y_ulb) in zip(dataloaders['train_sup'], dataloaders['train_unsup']):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)
        x_ulb, y_ulb = x_ulb.to(device), y_ulb.to(device)

        # Outputs of labeled data
        logits_lb = model(x_lb)

        # Supervised loss
        sup_loss = criterion(logits_lb, y_lb)

        # Outputs of unlabeled data but without batch norm
        bn_backup = freeze_bn(model)
        logits_ulb = model(x_ulb)
        unfreeze_bn(model, bn_backup)

        # Generate pseudo labels in form of argmax of probas
        ps_lb = generate_pseudo_labels(logits_ulb, use_hard_labels)

        # Generate a mask in Form of a boolean tensor that is true if argmax of probas is > p_cutoff
        mask = generate_mask(logits_ulb, p_cutoff)

        # Unsupervised Loss for highly certain pseudo labels
        unsup_loss = (F.cross_entropy(logits_ulb, ps_lb, reduction='none') * mask).mean()

        # Calculate SSL Warm Up Factor
        unsup_warmup_ = torch.clip(torch.tensor(epoch / (unsup_warmup * n_epochs)),  min=0.0, max=1.0)

        # Loss thats used for backpropagation
        total_loss = sup_loss + unsup_warmup_ * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(
            sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(), 
            total_loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"],
            mask_ratio=mask.float().mean().item()
            )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats