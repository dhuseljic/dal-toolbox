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

# @torch.inference_mode()
# def f1(output, target, num_classes, device):
#     _, pred = output.topk(1, 1, True, True)
#     pred = pred.t()
#     if num_classes <= 2:
#         f1_score = F1Score('binary', num_classes=num_classes, average='macro').to(device) 
#         f1_score = f1_score(pred.squeeze(0), target)
#     else:
#         f1_score = F1Score('multiclass', num_classes=num_classes, average='macro').to(device) 
#         f1_score = f1_score(pred.squeeze(0), target)
#     return f1_score

# !TODO: Sklearn and torchmetrics scores do not align 
def f1_macro(output, target, num_classes, device):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().squeeze(0)
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if num_classes <=2:
        f1 = f1_score(target, pred, average='binary')
    
    else:
        f1 = f1_score(target, pred, average='macro')

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
    auc_x = np.array(range(len(metric)))
    auc_y = np.array(metric)
    span_x = (auc_x[-1] - auc_x[0]) # taken from: revisiting uncertainty dalnlp 
    return (auc(auc_x, auc_y) / span_x).round(3)

def binary_stat_scores(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    tp = ((target == pred) & (target == 1)).sum([0,1]).squeeze()
    fn = ((target != pred) & (target == 1)).sum([0,1]).squeeze()
    fp = ((target != pred) & (target == 0)).sum([0,1]).squeeze()
    tn = ((target == pred) & (target == 0)).sum([0,1]).squeeze()
    return tp, fp, tn, fn

def multiclass_stat_scores(output, target, num_classes, average='micro'):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    if average == 'macro':
        unique_mapping = (target * num_classes + pred).to(torch.long)    
        bins = torch.bincount(unique_mapping)
        confmat = bins.reshape(num_classes, num_classes)
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
        return tp, fp, tn, fn
    
    elif average == 'micro':
        tp = (pred == target).sum()
        fp = (pred != target).sum()
        fn = (pred != target).sum()
        tn = num_classes * pred.numel() - (fp + fn + tp)
    
    return tp, fp, tn, fn
    




