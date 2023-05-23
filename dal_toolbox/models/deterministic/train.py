import torch.nn as nn

from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue


def train_one_epoch_bertmodel(model, dataloader, epoch, optimizer, scheduler, criterion, device, print_freq=25):
    # TODO(lrauch): please move this to trainer and remove, remove file pls
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
        if model.num_classes <= 2: 
            batch_f1_macro = generalization.f1_macro(logits, targets, 'binary')
            batch_f1_micro = batch_f1_macro
        else:
            batch_f1_macro = generalization.f1_macro(logits, targets, 'macro')
            batch_f1_micro = generalization.f1_macro(logits, targets, 'micro')

        batch_acc_balanced = generalization.balanced_acc(logits, targets)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)
        metric_logger.meters['batch_f1_macro'].update(batch_f1_macro.item(), n=batch_size)
        metric_logger.meters['batch_f1_micro'].update(batch_f1_micro.item(), n=batch_size)
        metric_logger.meters['batch_acc_balanced'].update(batch_acc_balanced.item(), n=batch_size)

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
def train_one_epoch_pseudolabel(model, dataloaders, criterion, optimizer, n_iter, p_cutoff, lambda_u, device,
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

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             mask_ratio=mask.float().mean().item(), unsup_warmup_factor=unsup_warmup_factor, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


def train_one_epoch_pimodel(model, dataloaders, criterion, optimizer, n_iter, lambda_u, device, unsup_warmup=.4,
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

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(loss=loss.item(), sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                             unsup_warmup_factor=unsup_warmup_factor, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


def train_one_epoch_fixmatch(model, dataloaders, criterion, optimizer, device, use_hard_labels,
                    lambda_u, p_cutoff, use_cat, T, epoch=None, print_freq=50):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    labeled_loader = dataloaders['train_sup']
    unlabeled_iter1 = iter(dataloaders['train_unsup_weak'])
    unlabeled_iter2 = iter(dataloaders['train_unsup_strong'])

    for x_lb, y_lb in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)
        x_ulb_weak, _ = next(unlabeled_iter1)
        x_ulb_weak = x_ulb_weak.to(device)
        x_ulb_strong, _ = next(unlabeled_iter2)
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
