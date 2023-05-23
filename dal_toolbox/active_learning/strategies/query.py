import numpy as np
from abc import ABC, abstractmethod


class Query(ABC):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        # set rng which should be used for all random stuff
        if random_seed is None:
            self.rng = np.random.mtrand._rand
        else:
            self.rng = np.random.RandomState(self.random_seed)

    @abstractmethod
    def query(self):
        pass
