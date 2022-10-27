import torch

from torch.utils.data import DataLoader
from metrics import ood
from .query import Query


class UncertaintySampling(Query):
    def __init__(self, uncertainty_type):
        super().__init__()
        self.uncertainty_type = uncertainty_type

    @torch.no_grad()
    def query(self, model, dataset, acq_size, batch_size, device):
        dataloader = DataLoader(dataset.unlabeled_dataset, batch_size=batch_size, shuffle=False)
        logits = model.forward_logits(dataloader, device)
        probas = logits.softmax(-1)

        if self.uncertainty_type == 'least_confident':
            score, _ = probas.min(dim=-1)
        elif self.uncertainty_type == 'entropy':
            score = ood.entropy_fn(probas)
        else:
            NotImplementedError(f"{self.uncertainty_type} is not implemented!")

        _, indices = score.topk(acq_size)
        actual_indices = [dataset.unlabeled_indices[i] for i in indices]
        return actual_indices
