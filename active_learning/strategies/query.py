import random
from abc import ABC, abstractmethod

class Query(ABC):
    def __init__(self, random_seed=42):
        # set rng which should be used for all random stuff
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)

    @abstractmethod
    def query(self):
        pass