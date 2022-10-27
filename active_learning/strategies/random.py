import random
from .query import Query

class RandomSampling(Query):
    def query(self, dataset, acq_size, **kwargs):
        # TODO some args should not be in here:
        # see https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/utils/_functions.py
        return random.sample(dataset.unlabeled_indices, acq_size)

