import torch
import numpy as np
from sklearn.metrics import precision_score

@torch.inference_mode()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if target.ndim == 2:
        target = target.max(dim=1)[1]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / batch_size))
    return res


def avg_precision(output, targets):
    _, pred = output.topk(1, 1, True, True)
    pred = np.array(pred)
    targets = np.array(targets)
    return precision_score(targets, pred, average='weighted')