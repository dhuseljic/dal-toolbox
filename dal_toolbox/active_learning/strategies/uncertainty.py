import torch

from abc import ABC, abstractmethod

from .query import Query
from ..data import ActiveLearningDataModule
from ...models.utils.base import BaseModule
from ...metrics import entropy_from_logits, entropy_from_probas, ensemble_log_softmax, ensemble_entropy_from_logits


class UncertaintySampling(Query, ABC):
    def __init__(self, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size

    @torch.no_grad()
    def query(
        self,
        *,
        model: BaseModule,
        al_datamodule: ActiveLearningDataModule,
        acq_size: int,
        return_utilities: bool = False,
        **kwargs
    ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader)
        scores = self.get_utilities(logits)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        if return_utilities:
            return actual_indices, scores
        return actual_indices

    @abstractmethod
    def get_utilities(self, logits):
        pass


class EntropySampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        return entropy_from_logits(logits)


class LeastConfidentSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        probas = logits.softmax(-1)
        scores, _ = probas.max(dim=-1)
        scores = 1 - scores
        return scores


class MarginSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 2-dimensional, got shape {logits.shape}")
        probas = logits.softmax(-1)
        top_probas, _ = torch.topk(probas, k=2, dim=-1)
        scores = top_probas[:, 0] - top_probas[:, 1]
        scores = 1 - scores
        return scores


class BayesianEntropySampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        return ensemble_entropy_from_logits(logits)


class BayesianLeastConfidentSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        log_probas = ensemble_log_softmax(logits)
        probas = log_probas.exp()
        scores, _ = probas.max(dim=-1)
        scores = 1 - scores
        return scores


class BayesianMarginSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        log_probas = ensemble_log_softmax(logits)
        probas = log_probas.exp()
        top_probas, _ = torch.topk(probas, k=2, dim=-1)
        scores = top_probas[:, 0] - top_probas[:, 1]
        scores = 1 - scores
        return scores


class VariationRatioSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        return self._variation_ratio(logits)

    def _variation_ratio(self, logits: torch.Tensor) -> torch.Tensor:
        # TODO(dhuseljic): should be in own file, so we can use it without the class
        """
        Computes the variation ratio for each sample in a Bayesian model. The variation ratio is
            the proportion of predicted class labels that are not the modal class prediction. 
            Specifically, it is computed as the number of ensemble members that give a different 
            prediction than the modal prediction, divided by the total number of ensemble members.

        Args:
            logits: A tensor of shape (n_samples, n_members, n_classes) representing the logits
                of a Bayesian model, where n_samples is the number of input samples, n_members is
                the number of members in the ensemble, and n_classes is the number of output classes.

        Returns:
            A tensor of shape (n_samples,) containing the variation ratio for each sample.

        Raises:
            ValueError: If logits tensor is not 3-dimensional.

        """
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        ensemble_size = logits.size(1)

        # Compute predicted classes and modal prediction for each sample
        pred_classes = logits.argmax(dim=-1)
        modal_preds, _ = torch.mode(pred_classes, dim=1)

        # Compute a binary mask indicating predictions that not match the modal one
        diff_mask = (pred_classes != modal_preds[:, None])

        # Compute the variation ratio
        num_diff = torch.sum(diff_mask, dim=1)
        var_ratio = num_diff.float() / ensemble_size

        return var_ratio


class BALDSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {logits.shape}")
        scores = self.bald_score(logits)
        return scores

    def bald_score(self, logits):
        # TODO(dhuseljic): implement bald from logits
        # probas = logits.softmax(-1)
        # ensemble_probas = torch.mean(probas, dim=1)
        # ensemble_entropy = entropy_from_probas(ensemble_probas)
        # mean_entropy = entropy_from_probas(probas).mean(dim=1)
        # score = ensemble_entropy - mean_entropy

        ensemble_entropy = ensemble_entropy_from_logits(logits)
        mean_entropy = entropy_from_probas(logits.softmax(-1)).mean(dim=1)
        score = ensemble_entropy - mean_entropy

        return score
