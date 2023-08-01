import numpy as np
import torch

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Subset

from .query import Query
from ..data import ActiveLearningDataModule
from ...models.utils.base import BaseModule
from ...metrics import entropy_from_logits, entropy_from_probas, ensemble_log_softmax, ensemble_entropy_from_logits


class MELL(Query, ABC):
    def __init__(self, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size

    @torch.no_grad()
    def query(self, *, model: BaseModule, al_datamodule: ActiveLearningDataModule, acq_size: int, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        val_set = al_datamodule.val_dataset  # TODO (ynagel) This should stem from the train dataset maybe?
        val_dataloader = DataLoader(
            Subset(val_set, np.arange(100)))

        unlabeled_probas = []

        for batch in unlabeled_dataloader:
            logits = model.mc_forward(batch[0])  # TODO (ynagel) How to make work normally?
            unlabeled_probas.append(torch.softmax(logits, dim=-1))

        unlabeled_probas = torch.cat(unlabeled_probas, dim=0)

        val_probas = []

        for batch in val_dataloader:
            logits = model.mc_forward(batch[0])  # TODO (ynagel) How to make work normally?
            val_probas.append(torch.softmax(logits, dim=-1))

        val_probas = torch.cat(val_probas, dim=0)
        n_val = val_probas.shape[0]

        pr_y_i_c = torch.mean(unlabeled_probas, dim=1)  # (n_samples, n_classes)

        pr_y_i_c_y_j_c_ = torch.zeros(
            size=(unlabeled_probas.shape[0], val_probas.shape[0], unlabeled_probas.shape[-1], val_probas.shape[-1]))
        # num samples unlabels x num samples validation x num classes x num classes
        # TODO (ynagel) Find efficient implementation

        print(pr_y_i_c_y_j_c_.shape)

        unlabeled_probas = torch.reshape(unlabeled_probas, (
            unlabeled_probas.shape[0], unlabeled_probas.shape[2], unlabeled_probas.shape[1]))
        val_probas = torch.reshape(val_probas, (val_probas.shape[0], val_probas.shape[2], val_probas.shape[1]))
        import time
        start = time.time()
        for i, y_i in enumerate(unlabeled_probas):
            for j, y_j in enumerate(val_probas):
                for c in range(unlabeled_probas.shape[-1]):
                    for c_star in range(val_probas.shape[-1]):
                        # print(i, j, c, c_star)
                        # print(y_i[c] * y_j[c_star])
                        pr_y_i_c_y_j_c_[i, j, c, c_star] = torch.sum(y_i[c] * y_j[c_star])
                        # print(pr_y_i_c_y_j_c_[i, j, c, c_star])

        print(f"Took {time.time() - start} seconds.")

        h_y_i = - torch.sum(torch.special.entr(pr_y_i_c), dim=-1)
        h_yi_y_j = -torch.sum(torch.sum(pr_y_i_c_y_j_c_ * torch.log2(pr_y_i_c_y_j_c_), dim=-1), dim=-1)
        scores = n_val * h_y_i - torch.sum(h_yi_y_j, dim=-1)

        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices
