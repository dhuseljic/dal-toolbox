import numpy as np
from abc import ABC, abstractmethod
from ...utils import setup_rng


class Query(ABC):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        # set rng which should be used for all random stuff
        self.rng = setup_rng(random_seed)

    @abstractmethod
    def query(self):
        pass
