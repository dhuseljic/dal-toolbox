import torch

from ..utils.pseudo_labels import freeze_bn, unfreeze_bn
from ..utils.pimodel import consistency_loss
from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue


def train_one_epoch(model, dataloaders, criterion, optimizer, device, n_epochs, lambda_u, unsup_warmup=.4,
                    epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    labeled_loader = dataloaders['train_sup']
    unlabeled_iter1 = iter(dataloaders['train_unsup_weak_1'])
    unlabeled_iter2 = iter(dataloaders['train_unsup_weak_2'])

    for x_lb, y_lb in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)

        # Outputs of labeled data
        logits_lb = model(x_lb)
        sup_loss = criterion(logits_lb, y_lb)

        x_ulb_weak_1, _ = next(unlabeled_iter1)
        x_ulb_weak_1 = x_ulb_weak_1.to(device)
        x_ulb_weak_2, _ = next(unlabeled_iter2)
        x_ulb_weak_2 = x_ulb_weak_2.to(device)

        # Outputs of unlabeled data but without batch norm
        bn_backup = freeze_bn(model)
        logits_ulb_weak_1 = model(x_ulb_weak_1)
        logits_ulb_weak_2 = model(x_ulb_weak_2)
        unfreeze_bn(model, bn_backup)

        unsup_loss = consistency_loss(logits_ulb_weak_2, torch.softmax(logits_ulb_weak_1.detach(), dim=-1), 'mse')
        # Calculate SSL Warm Up Factor
        unsup_warmup_ = torch.clip(torch.tensor(epoch / (unsup_warmup * n_epochs)),  min=0.0, max=1.0)

        # Loss thats used for backpropagation
        loss = sup_loss + unsup_warmup_ * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats
