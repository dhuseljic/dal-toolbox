import torch
import numpy as np


def mixup(inputs: torch.Tensor, targets_one_hot: torch.Tensor, mixup_alpha: float):
    lmb = np.random.beta(mixup_alpha, mixup_alpha)
    indices = torch.randperm(len(inputs), device=inputs.device, dtype=torch.long)
    inputs_mixed = lmb * inputs + (1 - lmb) * inputs[indices]
    targets_mixed = lmb * targets_one_hot + (1 - lmb) * targets_one_hot[indices]
    return inputs_mixed, targets_mixed
