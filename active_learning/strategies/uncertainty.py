import torch

from torch.utils.data import DataLoader
from metrics import ood
from .query import Query


class UncertaintySampling(Query):
    def __init__(self, batch_size=128, uncertainty_type='entropy', subset_size=None, device='cuda'):
        super().__init__()
        self.uncertainty_type = uncertainty_type
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device


    @torch.no_grad()
    def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
        del kwargs
        if not hasattr(model, 'get_probas'):
            raise ValueError('The method `get_probas` is mandatory to use uncertainty sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)
        probas = model.get_probas(dataloader, device=self.device)

        if self.uncertainty_type == 'least_confident':
            scores, _ = probas.min(dim=-1)
        elif self.uncertainty_type == 'entropy':
            scores = ood.entropy_fn(probas)
        else:
            NotImplementedError(f"{self.uncertainty_type} is not implemented!")

        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices