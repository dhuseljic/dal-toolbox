import torch
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


def generate_mask(logits_x_ulb, p_cutoff):
    """
        Returns: A boolean tensor that is True where input is greater than or equal to p_cutoff and False elsewhere
    """
    probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
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