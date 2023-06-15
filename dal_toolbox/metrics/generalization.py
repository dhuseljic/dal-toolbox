import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


class Accuracy(torchmetrics.Metric):
    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk
        self.add_state('num_correct', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('num_samples', torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        batch_size = len(logits)
        if logits.shape == targets.shape:
            targets = self._convert_one_hot_targets(targets)

        if self.topk > 1:
            _, class_preds = logits.topk(self.topk)
            self.num_correct += (class_preds.T == targets[None]).float().sum()
        else:
            class_preds = logits.argmax(-1)
            self.num_correct += (class_preds == targets).float().sum()
        self.num_samples += batch_size

    def compute(self):
        return self.num_correct / self.num_samples

    def _convert_one_hot_targets(self, targets):
        return targets.argmax(-1)


class ContrastiveAccuracy(torchmetrics.Metric):
    """
    A ``torchmetrics.Metric`` for calculating the contrastive accuracy.

    The contrastive accuracy describes how accurately a constructive model can distinguish the positive pairs from the
    negative pairs inside a batch.
    """

    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk
        self.add_state('num_correct', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('num_samples', torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        cos_sim = F.cosine_similarity(logits[:, None, :], logits[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        comb_sim = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        if self.topk > 1:
            self.num_correct += (sim_argsort < self.topk).float().sum()
        else:
            self.num_correct += (sim_argsort == 0).float().sum()

        self.num_samples += len(logits)

    def compute(self):
        return self.num_correct / self.num_samples


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
def f1_macro(output, target, mode):
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

    return f1

def balanced_acc(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().squeeze(0)
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return balanced_accuracy_score(target, pred)


def avg_precision(output, targets):
    _, pred = output.topk(1, 1, True, True)
    pred = np.array(pred)
    targets = np.array(targets)
    return precision_score(targets, pred, average='weighted', zero_division=0)


def area_under_curve(metric):
    auc_x = np.arange(len(metric))
    auc_y = np.array(metric)
    span_x = (auc_x[-1] - auc_x[0])  # taken from: revisiting uncertainty dalnlp
    return (auc(auc_x, auc_y) / span_x).round(4)
