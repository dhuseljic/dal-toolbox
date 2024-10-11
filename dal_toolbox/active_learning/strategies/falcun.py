from .query import Query

import time
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
import torch
from sklearn.preprocessing import MinMaxScaler


class Falcun(Query):
    def __init__(self, subset_size, gamma=10, deterministic=False, custom_dist="distance"):
        super().__init__()
        self.subset_size = subset_size
        self.deterministic = deterministic
        self.gamma = gamma
        self.custom_dist = custom_dist


    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        loader_unlabeled, idxs_unlabeled = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        probs = model.model.get_probas(dataloader=loader_unlabeled, device='cuda')
        unc = self.get_unc(probs, uncertainty="margin")

        if self.deterministic and self.custom_dist == "unc":
            _, idx = unc.sort(descending=True)
            sel_ids = idx[:acq_size]
        else:
            sel_ids = self.get_indices(acq_size, probs.cpu().numpy(), unc.cpu().numpy())

        return [idxs_unlabeled[idx] for idx in sel_ids]


    def get_unc(self, probs, uncertainty="margin"):
        probs_sorted, _ = probs.sort(descending=True)
        if uncertainty == "margin":
            return 1 - (probs_sorted[:, 0] - probs_sorted[:, 1])
        elif uncertainty == "entropy":
            probs += 1e-8
            entropy = -(probs * torch.log(probs)).sum(1)
            return (entropy - entropy.min()) / (entropy.max() - entropy.min())
        else:  # lc
            return 1 - probs_sorted[:, 0]

    def update_gamma(self):
        self.gamma += 1

    def get_indices(self, n, probs, unc):
        ind_selected = []
        vec_selected = []

        dists = unc.copy()
        unlabeled_range = np.arange(len(dists))
        candidate_mask = np.ones(len(dists), dtype=bool)

        while len(vec_selected) < n:
            if len(vec_selected) > 0:
                new_dists = pairwise_distances(probs, [vec_selected[-1]], metric="l1").ravel().astype(float)
                dists = np.array([dists[i] if dists[i] < new_dists[i] else new_dists[i] for i in range(len(probs))])
            if self.deterministic:
                if self.custom_dist == "distance":
                    ind = dists.argmax()
                elif self.custom_dist == "distance_unc_norm":
                    scaler = MinMaxScaler((0, 1))
                    scaler.fit(dists[candidate_mask].reshape(-1, 1))
                    _dists = scaler.transform(dists.reshape(-1, 1)).ravel()
                    _rel = _dists + unc
                    ind = _rel.argmax()
            else:
                if sum(dists[candidate_mask]) > 0:

                    if self.custom_dist == "distance_unc_norm":
                        scaler = MinMaxScaler((0, 1))
                        scaler.fit(dists[candidate_mask].reshape(-1, 1))
                        _dists = scaler.transform(dists[candidate_mask].reshape(-1, 1)).ravel()
                        dist_probs = (_dists + unc[candidate_mask]) ** self.gamma / sum((_dists + unc[candidate_mask]) ** self.gamma)
                    elif self.custom_dist == "distance":
                        dist_probs = dists[candidate_mask] ** self.gamma / sum(dists[candidate_mask] ** self.gamma)
                    elif self.custom_dist == "unc":
                        dist_probs = unc[candidate_mask] ** self.gamma / sum(unc[candidate_mask] ** self.gamma)
                    else:
                        raise NotImplementedError

                    ind = np.random.choice(unlabeled_range[candidate_mask], size=1, p=dist_probs)[0]
                else:
                    ind = np.random.choice(unlabeled_range[candidate_mask], size=1)[0]
            candidate_mask[ind] = False
            vec_selected.append(probs[ind])
            ind_selected.append(ind)
        return ind_selected