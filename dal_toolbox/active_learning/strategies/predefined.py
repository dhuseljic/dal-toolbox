import json
import numpy as np

from .query import Query


class PredefinedSampling(Query):
    def __init__(self, queried_indices_json, n_acq, n_init, acq_size):
        super().__init__()
        self.queried_indices_json = queried_indices_json
        self.n_acq = n_acq
        self.n_init = n_init
        self.acq_size = acq_size

        # load indices
        with open(self.queried_indices_json, 'r', encoding='utf-8') as f:
            queried_indices = json.load(f)

        # Verify number of acquisitions
        n_acq_json = len(queried_indices) - 1
        if self.n_acq != n_acq_json:
            raise ValueError(f'Number of acquisitions ({self.n_acq}) differs from the one in json file ({n_acq_json}).')

        # Verify number of samples in initial pool
        n_init_json = len(queried_indices['cycle0'])
        if self.n_init != n_init_json:
            raise ValueError(f'Initial pool size ({self.n_init}) differs from the one in json file ({n_init_json}).')

        # Verify acquisition size
        acq_size_json = len(queried_indices['cycle1'])
        if self.acq_size != acq_size_json:
            raise ValueError(f'Acquisition size ({self.acq_size}) differs from the one in json file ({acq_size_json}).')

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
