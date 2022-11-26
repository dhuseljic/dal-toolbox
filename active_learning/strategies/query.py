import torch
from abc import ABC, abstractmethod

class Query(ABC):
    def __init__(self, random_seed=42, device=None):
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(random_seed)

    @abstractmethod
    def query(self):
        pass