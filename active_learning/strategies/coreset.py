import torch
import numpy as np
from sklearn.metrics import pairwise_distances

from torch.utils.data import DataLoader
from .query import Query


class CoreSet(Query):
    def __init__(self, subset_size=None):
        super().__init__()
        self.subset_size = subset_size

    def kcenter_greedy(self, features_unlabeled: torch.Tensor, features_labeled: torch.Tensor, acq_size: int):
        features_labeled = features_labeled.numpy()
        features_unlabeled = features_unlabeled.numpy()

        n_unlabeled = len(features_unlabeled)

        distances = pairwise_distances(features_unlabeled, features_labeled)
        min_dist = np.min(distances, axis=1)

        idxs = []
        for _ in range(acq_size):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(features_unlabeled, features_unlabeled[[idx]])
            for j in range(n_unlabeled):
                min_dist[j] = np.min((min_dist[j], dist_new_ctr[j, 0]))

        return idxs

    def query(self, model, al_dataset, acq_size, batch_size, device):
        if not hasattr(model, 'get_representation'):
            raise ValueError('The method `get_representation` is mandatory to use core set sampling.')

        unlabeled_indices = al_dataset.unlabeled_indices
        labeled_indices = al_dataset.labeled_indices
        query_dataset = al_dataset.query_dataset

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)

        unlabeled_dataloader = DataLoader(query_dataset, batch_size=batch_size, sampler=unlabeled_indices)
        features_unlabeled = model.get_representation(unlabeled_dataloader, device)

        labeled_dataloader = DataLoader(query_dataset, batch_size=batch_size, sampler=labeled_indices)
        features_labeled = model.get_representation(labeled_dataloader, device)

        chosen = self.kcenter_greedy(features_unlabeled, features_labeled, acq_size)
        return [unlabeled_indices[idx] for idx in chosen]
