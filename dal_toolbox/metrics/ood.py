import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import torchmetrics


def ood_aupr(score_id: torch.Tensor, score_ood: torch.Tensor):
    """Computes the AUROC and assumes that a higher score is OOD.

    Args:
        score_id (torch.Tensor): Score of in-distribution samples.
        score_ood (torch.Tensor): Score of out-of-distribution samples.

    Returns:
        float: The AUROC score.
    """
    y_true = torch.cat((torch.zeros(len(score_id)), torch.ones(len(score_ood))))
    y_score = torch.cat((score_id, score_ood))
    return average_precision_score(y_true, y_score)


def ood_auroc(score_id: torch.Tensor, score_ood: torch.Tensor):
    """Computes the AUROC and assumes that a higher score is OOD.

    Args:
        score_id (torch.Tensor): Score of in-distribution samples.
        score_ood (torch.Tensor): Score of out-of-distribution samples.

    Returns:
        float: The AUROC score.
    """
    y_true = torch.cat((torch.zeros(len(score_id)), torch.ones(len(score_ood))))
    y_score = torch.cat((score_id, score_ood))
    return roc_auc_score(y_true, y_score)


class OODAUROC(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('scores_id', default=[], dist_reduce_fx='cat')
        self.add_state('scores_ood', default=[], dist_reduce_fx='cat')

    def update(self, scores_id: torch.Tensor, scores_ood: torch.Tensor):
        self.scores_id.append(scores_id)
        self.scores_ood.append(scores_ood)

    def compute(self):
        scores_id = torch.cat(self.scores_id)
        scores_ood = torch.cat(self.scores_ood)

        preds = torch.cat((scores_id, scores_ood))
        targets = torch.cat((torch.zeros(len(scores_id)), torch.ones(len(scores_ood))))
        targets = targets.long()

        auroc = torchmetrics.functional.auroc(preds, targets, task='binary')
        return auroc


class OODAUPR(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('scores_id', default=[], dist_reduce_fx='cat')
        self.add_state('scores_ood', default=[], dist_reduce_fx='cat')

    def update(self, scores_id: torch.Tensor, scores_ood: torch.Tensor):
        self.scores_id.append(scores_id)
        self.scores_ood.append(scores_ood)

    def compute(self):
        scores_id = torch.cat(self.scores_id)
        scores_ood = torch.cat(self.scores_ood)

        preds = torch.cat((scores_id, scores_ood))
        targets = torch.cat((torch.zeros(len(scores_id)), torch.ones(len(scores_ood))))

        aupr = torchmetrics.functional.average_precision(preds, targets.long(), task='binary')
        return aupr
