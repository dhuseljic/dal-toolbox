from .query import Query

# TODO some args should not be in here:
# see https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/utils/_functions.py


class RandomSampling(Query):
    def query(self, *, al_datamodule, acq_size, **kwargs) -> list:
        unlabeled_indices = al_datamodule.unlabeled_indices
        indices = self.rng.choice(unlabeled_indices, size=acq_size, replace=False)
        indices = indices.tolist()
        return indices
