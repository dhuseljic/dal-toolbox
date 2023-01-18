import torch
import torch.nn.functional as F
import torch.nn as nn

def generate_pseudo_labels(logits, use_hard_label=True, T=1.0, softmax=True):
        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            return pseudo_label
        
        # return soft label
        if softmax:
            pseudo_label = torch.softmax(logits / T, dim=-1)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label


def generate_mask(logits_x_ulb, p_cutoff, softmax=True):
    """
        Returns: A boolean tensor that is True where input is greater than or equal to p_cutoff and False elsewhere
    """
    if softmax:
        probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
    else:
        probs_x_ulb = logits_x_ulb
    max_probs, _ = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(p_cutoff)
    return mask


def freeze_bn(model):
    backup = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.SyncBatchNorm) or isinstance(module, nn.BatchNorm2d):
            backup[name + '.running_mean'] = module.running_mean.data.clone()
            backup[name + '.running_var'] = module.running_var.data.clone()
            backup[name + '.num_batches_tracked'] = module.num_batches_tracked.data.clone()
    return backup


def unfreeze_bn(model, backup):
    for name, module in model.named_modules():
        if isinstance(module, nn.SyncBatchNorm) or isinstance(module, nn.BatchNorm2d):
            module.running_mean.data = backup[name + '.running_mean']
            module.running_var.data = backup[name + '.running_var']
            module.num_batches_tracked.data = backup[name + '.num_batches_tracked']


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.
    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()