import torch

from torch.utils.data import DataLoader

from .query import Query


class CoreSet(Query):
    def __init__(self, subset_size=None, batch_size=128, device='cuda'):
        super().__init__()
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    def kcenter_greedy(self, features_unlabeled: torch.Tensor, features_labeled: torch.Tensor, acq_size: int):
        n_unlabeled = len(features_unlabeled)

        distances = torch.cdist(features_unlabeled, features_labeled)
        min_dist, _ = torch.min(distances, axis=1)

        idxs = []
        for _ in range(acq_size):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = torch.cdist(features_unlabeled, features_unlabeled[idx].unsqueeze(0))
            for j in range(n_unlabeled):
                min_dist[j] = torch.min(min_dist[j], dist_new_ctr[j, 0])
        return idxs

    def query(self, model, dataset, unlabeled_indices, labeled_indices, acq_size, **kwargs):
        if not hasattr(model, 'get_representation'):
            raise ValueError('The method `get_representation` is mandatory to use core set sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)

        if "collator" in list(kwargs.keys()):
            unlabeled_dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size*2, 
                collate_fn=kwargs['collator'],
                sampler=unlabeled_indices)
            labeled_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size*2, 
                collate_fn=kwargs['collator'],
                sampler=labeled_indices)
        else:
            unlabeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)
            labeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=labeled_indices)
        del kwargs

        features_unlabeled = model.get_representation(unlabeled_dataloader, self.device)
        features_labeled = model.get_representation(labeled_dataloader, self.device)

        chosen = self.kcenter_greedy(features_unlabeled, features_labeled, acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
