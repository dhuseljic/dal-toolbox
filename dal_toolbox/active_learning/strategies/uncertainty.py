import torch

from torch.utils.data import DataLoader

from .query import Query
from ...metrics import ood
from abc import ABC, abstractmethod


class UncertaintySampling(Query, ABC):
    def __init__(self, batch_size=64, subset_size=None, device='cuda', random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
        if not hasattr(model, 'get_probas'):
            raise ValueError('The method `get_probas` is mandatory to use uncertainty sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.choice(unlabeled_indices, size=self.subset_size, replace=False)
            unlabeled_indices = unlabeled_indices.tolist()

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                sampler=unlabeled_indices, collate_fn=kwargs.get("collator"))

        probas = model.get_probas(dataloader, device=self.device)
        scores = self.get_scores(probas)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices

    @abstractmethod
    def get_scores(self, probas):
        pass


class EntropySampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 2:
            raise ValueError(f"Input probas tensor must be 2-dimensional, got shape {probas.shape}")
        return ood.entropy_fn(probas)


class LeastConfidentSampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 2:
            raise ValueError(f"Input probas tensor must be 2-dimensional, got shape {probas.shape}")
        scores, _ = probas.max(dim=-1)
        scores = 1 - scores
        return scores


class MarginSampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 2:
            raise ValueError(f"Input probas tensor must be 2-dimensional, got shape {probas.shape}")
        top_probas, _ = torch.topk(probas, k=2, dim=-1)
        scores = top_probas[:, 0] - top_probas[:, 1]
        scores = 1 - scores
        return scores


class BayesianEntropySampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {probas.shape}")
        probas = torch.mean(probas, dim=-1)
        return ood.entropy_fn(probas)


class BayesianLeastConfidentSampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {probas.shape}")
        probas = torch.mean(probas, dim=-1)
        scores, _ = probas.max(dim=-1)
        scores = 1 - scores
        return scores


class BayesianMarginSampling(UncertaintySampling):
    def get_scores(self, probas):
        if probas.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {probas.shape}")
        probas = torch.mean(probas, dim=-1)
        top_probas, _ = torch.topk(probas, k=2, dim=-1)
        scores = top_probas[:, 0] - top_probas[:, 1]
        scores = 1 - scores
        return scores


class VariationRatioSampling(UncertaintySampling):

    def _variation_ratio(self, logits: torch.Tensor) -> torch.Tensor:
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
        # TODO: should be in own file, so we can use it without the class
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        n_member = logits.size(1)

        preds_classes = logits.argmax(dim=-1)

        # Compute the modal prediction for each sample
        modal_preds, _ = torch.mode(preds_classes, dim=1)

        # Compute a binary mask indicating different predictions
        diff_mask = (preds_classes != modal_preds.unsqueeze(dim=1))

        # Compute the variation ratio
        num_diff = torch.sum(diff_mask, dim=1)
        var_ratio = num_diff.float() / n_member

        return var_ratio

    def get_scores(self, probas):
        if probas.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {probas.shape}")
        # TODO(dhuseljic): change to logits?
        return self._variation_ratio(probas)


class BALDSampling(UncertaintySampling):

    def bald_score(self, probas):
        mean_probas = torch.mean(probas, dim=1)
        mean_entropy = ood.entropy_fn(mean_probas)
        entropy = ood.entropy_fn(probas).mean(-1)
        score = mean_entropy - entropy
        return score

    def get_scores(self, probas):
        if probas.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {probas.shape}")
        scores = self.bald_score(probas)
        return scores
