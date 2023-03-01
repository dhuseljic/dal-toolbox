import torch
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

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

# !TODO: Sklearn and torchmetrics scores do not align 
def f1_macro(output, target, mode, device):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().squeeze(0)
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if mode == 'binary':
        f1 = f1_score(target, pred, average='binary')
    elif mode =='macro':
        f1 = f1_score(target, pred, average='macro')
    elif mode == 'micro':
        f1 = f1_score(target, pred, average='micro')

    return torch.from_numpy(np.asarray(f1)).to(device)

def balanced_acc(output, target, device):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().squeeze(0)
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    return torch.from_numpy(np.asarray(balanced_accuracy_score(target, pred))).to(device)

def avg_precision(output, targets):
    _, pred = output.topk(1, 1, True, True)
    pred = np.array(pred)
    targets = np.array(targets)
    return precision_score(targets, pred, average='weighted', zero_division=0)

def area_under_curve(metric):
    auc_x = np.arange(len(metric))
    auc_y = np.array(metric)
    span_x = (auc_x[-1] - auc_x[0]) # taken from: revisiting uncertainty dalnlp 
    return (auc(auc_x, auc_y) / span_x).round(3)






