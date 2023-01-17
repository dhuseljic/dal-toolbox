import torch

from ..utils.pseudo_labels import generate_mask, generate_pseudo_labels
from ..utils.pimodel import consistency_loss
from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue


def train_one_epoch(model, dataloaders, criterion, optimizer, device, use_hard_labels,
                    lambda_u, p_cutoff, T):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    for (x_lb, y_lb), (x_ulb_weak, _), (x_ulb_strong, _) in zip(dataloaders['train_sup'], dataloaders['train_unsup_weak'], dataloaders['train_unsup_strong']):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)
        x_ulb_weak = x_ulb_weak.to(device)
        x_ulb_strong = x_ulb_strong.to(device)

        # Outputs of labeled data
        logits_lb = model(x_lb)

        # Supervised loss
        sup_loss = criterion(logits_lb, y_lb)

        # Outputs of strongly augmented data
        logits_ulb_strong = model(x_ulb_strong)

        # Outputs of weakly augmented data that should not influence the model-parameters
        with torch.no_grad():
            logits_ulb_weak = model(x_ulb_weak)

        # Generate mask
        mask = generate_mask(logits_ulb_weak, p_cutoff)

        # Generate pseudolabels
        ps_lb = generate_pseudo_labels(logits_ulb_weak, use_hard_labels, T=T)

        # Unsupervised Loss
        unsup_loss = consistency_loss(logits_ulb_strong,
                                          ps_lb,
                                          'ce',
                                          mask=mask)

        # Loss thats used for backpropagation
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