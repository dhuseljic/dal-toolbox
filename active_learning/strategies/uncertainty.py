import torch

from torch.utils.data import DataLoader
from metrics import ood
from .query import Query


class UncertaintySampling(Query):
    def __init__(self, uncertainty_type):
        super().__init__()
        self.uncertainty_type = uncertainty_type

    @torch.no_grad()
    def query(self, model, al_dataset, acq_size, batch_size, device):
        if not hasattr(model, 'get_probas'):
            raise ValueError('The method `get_probas` is mandatory to use uncertainty sampling.')
        dataloader = DataLoader(al_dataset.unlabeled_dataset, batch_size=batch_size)
        probas = model.get_probas(dataloader, device)

        if self.uncertainty_type == 'least_confident':
            scores, _ = probas.min(dim=-1)
        elif self.uncertainty_type == 'entropy':
            scores = ood.entropy_fn(probas)
        else:
            NotImplementedError(f"{self.uncertainty_type} is not implemented!")

        _, indices = scores.topk(acq_size)
        actual_indices = [al_dataset.unlabeled_indices[i] for i in indices]
        return actual_indices
