import json
import numpy as np

from .query import Query


class PredefinedSampling(Query):
    def __init__(self, queried_indices_json):
        super().__init__()
        self.queried_indices_json = queried_indices_json

        # load indices
        with open(self.queried_indices_json, 'r', encoding='utf-8') as f:
            queried_indices = json.load(f)

        self.indices = []
        for key in queried_indices:
            if key == 'cycle0':
                continue
            indices = queried_indices[key]
            self.indices.append(indices)

    def query(self, labeled_indices, acq_size, **kwargs):
        del kwargs
        chosen = self.indices.pop(0)

        if acq_size != len(chosen):
            raise ValueError(
                f'Wrong `acq_size` of {acq_size}. Experiment {self.queried_indices_json} used {len(chosen)}.')

        if np.any(np.isin(chosen, labeled_indices)):
            raise ValueError('Some chosen indices are already in the labeled pool.')

        return chosen
