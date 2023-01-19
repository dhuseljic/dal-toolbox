import torch
import torch.nn.functional as F

from ..utils import freeze_bn, unfreeze_bn
from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue


def train_one_epoch(model, dataloaders, criterion, optimizer, device, n_epochs, p_cutoff, lambda_u,
                    use_hard_labels=True, unsup_warmup=.4, epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    labeled_loader = dataloaders['train_sup']
    unlabeled_loader = dataloaders['train_unsup']
    unlabeled_iter = iter(unlabeled_loader)

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
        if use_hard_labels:
            probas_ulb = torch.softmax(logits_ulb.detach(), dim=-1)
        else:
            T = 1.0
            probas_ulb = torch.softmax(logits_ulb.detach()/T, dim=-1)
        max_probas, pseudo_label = torch.max(probas_ulb, dim=-1)
        mask = max_probas.ge(p_cutoff)

        unsup_loss = (F.cross_entropy(logits_ulb, pseudo_label, reduction='none') * mask).mean()
        unsup_warmup = torch.clip(torch.tensor(epoch / (unsup_warmup * n_epochs)),  min=0.0, max=1.0)

        # Loss thats used for backpropagation
        loss = sup_loss + unsup_warmup * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             mask_ratio=mask.float().mean().item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats
