import math
import torch


def clamp_probas(probas):
    eps = torch.finfo(probas.dtype).eps
    return probas.clamp(min=eps, max=1 - eps)


def entropy_from_probas(probas, dim=-1):
    probas = clamp_probas(probas)
    return - torch.sum(probas * probas.log(), dim=dim)


def entropy_from_logits(logits, dim=-1):
    # numerical stable version
    if logits.ndim != 2:
        raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
    log_probas = torch.log_softmax(logits, dim=dim)
    probas = log_probas.exp()
    entropy = - torch.sum(probas * log_probas, dim=dim)
    return entropy


def ensemble_log_softmax(logits, dim=-1, ensemble_dim=1):
    if logits.ndim != 3:
        raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
    ensemble_size = logits.size(ensemble_dim)
    # numerical stable version of avg ensemble probas: log sum_m^M exp log probs_m - log M = log 1/M sum_m probs_m
    log_probas = torch.logsumexp(logits.log_softmax(dim), dim=ensemble_dim) - math.log(ensemble_size)
    return log_probas


def ensemble_entropy_from_logits(logits, dim=-1, ensemble_dim=1):
    # numerical stable version
    if logits.ndim != 3:
        raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
    # numerical stable version of avg ensemble probas: log sum_m^M exp log probs_m - log M = log 1/M sum_m probs_m
    log_probas = ensemble_log_softmax(logits, dim=dim, ensemble_dim=ensemble_dim)
    probas = log_probas.exp()
    entropy = - torch.sum(probas * log_probas, dim=dim)
    return entropy


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


def log_sub_exp(x, y):
    larger = torch.max(x, y)
    smaller = torch.min(x, y)
    zero = torch.zeros_like(larger)
    result = larger + log1mexp(torch.max(larger - smaller, zero))
    return result


def log1mexp(x):
    x = torch.abs(x)
    return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))
