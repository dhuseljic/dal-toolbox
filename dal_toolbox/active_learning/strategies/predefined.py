import json
import numpy as np

from .query import Query


class PredefinedSampling(Query):
    def __init__(self, result_json):
        super().__init__()
        self.result_json = result_json

        # load indices
        with open(self.result_json, 'r') as f:
            loaded_results = json.load(f)

        self.indices = []
        for key in loaded_results:
            if key =='cycle0':
                continue
            cycle_results = loaded_results[key]
            self.indices.append(cycle_results['query_indices'])

    def query(self, labeled_indices, acq_size, **kwargs):
        del kwargs
        chosen = self.indices.pop(0)

        if acq_size != len(chosen):
            raise ValueError(f'Wrong `acq_size` of {acq_size}. Experiment {self.result_json} used {len(chosen)}.' )
        
        if np.any(np.isin(chosen, labeled_indices)):
            raise ValueError('Some chosen indices are already in the labeled pool.' )

        return chosen
