from .query import Query

# TODO some args should not be in here:
# see https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/utils/_functions.py


class RandomSampling(Query):
    def query(self, unlabeled_indices, acq_size, **kwargs):
        del kwargs  # del unused kwargs
        indices = self.rng.choice(unlabeled_indices, size=acq_size, replace=False)
        return indices
