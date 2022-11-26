import torch
from .query import Query

# TODO some args should not be in here:
# see https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/utils/_functions.py


class RandomSampling(Query):
    def query(self, dataset, acq_size, **kwargs):
        del kwargs  # del unused kwargs
        indices = torch.randperm(len(dataset.unlabeled_indices), generator=self.rng)[:acq_size]
        return dataset.unlabeled_indices[indices]
