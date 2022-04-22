import torch
from sklearn.metrics import roc_auc_score


def entropy_fn(probas, delta=1e-9):
    probas = probas.clone()
    probas = probas.clamp(min=delta, max=(1-delta))
    return - torch.sum(probas * probas.log(), -1)

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
