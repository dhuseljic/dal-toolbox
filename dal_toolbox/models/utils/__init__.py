import torch.nn as nn
from . import lr_scheduler


def freeze_bn(model):
    """Freezes the existing batch_norm layers in a module.

    Args:
        model (nn.Module): Deep neural network with batch norm layers.

    Returns:
        dict: Returns a dictionary of the tracked batchnorm statistics such as `running_mean` or `running_var`.
    """
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
