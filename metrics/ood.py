import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def clamp_probas(probas):
    eps = torch.finfo(probas.dtype).eps
    return probas.clamp(min=eps, max=1 - eps)


def entropy_fn(probas):
    probas = clamp_probas(probas)
    return - torch.sum(probas * probas.log(), -1)


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


def dempster_shafer_uncertainty(logits):
    """Defines the Dempster-Shafer Uncertainty for output logits.
    Under the Dempster-Shafer (DS) formulation of a multi-class model, the
    predictive uncertainty can be assessed as K/(K + sum(exp(logits))).
    This uncertainty metric directly measure the magnitude of the model logits,
    and is more properiate for a model that directly trains the magnitude of
    logits and uses this magnitude to quantify uncertainty (e.g., [1]).
    See Equation (1) of [1] for full detail.

    Args:
      logits (torch.Tensor): logits of model prediction (batch_size, num_classes).

    Returns:
      torch.Tensor: DS uncertainty estimate
    """
    num_classes = logits.shape[-1]
    belief_mass = torch.sum(torch.exp(logits), dim=-1)
    return num_classes / (belief_mass + num_classes)
