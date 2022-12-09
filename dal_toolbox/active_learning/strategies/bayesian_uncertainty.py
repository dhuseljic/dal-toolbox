import torch

from torch.utils.data import DataLoader
from .query import Query

class BayesianUncertaintySampling(Query):
    def __init__(self, uncertainty_type, n_mc_passes):
        super().__init__()
        self.uncertainty_type = uncertainty_type
        self.n_mc_passes = n_mc_passes

    @torch.no_grad()
    def query(self, model, dataset, acq_size, batch_size, device):
        dataloader = DataLoader(dataset.unlabeled_dataset, batch_size=batch_size, shuffle=False)
        mc_logits = model.get_mc_logits(dataloader, self.n_mc_passes, device)
        logits = torch.mean(mc_logits, dim=-1)
        probas = logits.softmax(dim=-1)

        if self.uncertainty_type == 'lc':
            meassurements = probas
        elif self.uncertainty_type == 'entropy':
            meassurements = - torch.sum(probas * probas.log(), dim=-1)
        else:
            NotImplementedError(f"{self.uncertainty_type} is not implemented!")
        
        _, indices = meassurements.topk(acq_size)
        actual_indices = [dataset.unlabeled_indices[i] for i in indices]
        return actual_indices
