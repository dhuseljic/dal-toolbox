import torch

from torch.utils.data import DataLoader

from .query import Query
from ...metrics import ood


class UncertaintySampling(Query):
    def __init__(self, batch_size=128, uncertainty_type='entropy', subset_size=None, device='cuda'):
        super().__init__()
        self.uncertainty_type = uncertainty_type
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    def get_scores(self, probas):
        if self.uncertainty_type == 'least_confident':
            scores, _ = probas.min(dim=-1)
        elif self.uncertainty_type == 'entropy':
            scores = ood.entropy_fn(probas)
        else:
            NotImplementedError(f"{self.uncertainty_type} is not implemented!")
        return scores

    @torch.no_grad()
    def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
        #TODO: Wieso del kwargs und wie mitnahme von collator
        if not hasattr(model, 'get_probas'):
            raise ValueError('The method `get_probas` is mandatory to use uncertainty sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)

        if "collator" in list(kwargs.keys()):
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size*2, 
                collate_fn=kwargs['collator'],
                sampler=unlabeled_indices)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)

        del kwargs
        probas = model.get_probas(dataloader, device=self.device)
        scores = self.get_scores(probas)

        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices

class CertaintySampling(UncertaintySampling):
    def get_scores(self, probas):
        return -super().get_scores(probas)