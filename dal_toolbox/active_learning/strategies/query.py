import random
import numpy as np
from abc import ABC, abstractmethod

class Query(ABC):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

        # set rng which should be used for all random stuff
        self.rng = random.Random(self.random_seed)
        self.np_rng = np.random.RandomState(self.random_seed)

    @abstractmethod
    def query(self):
        pass