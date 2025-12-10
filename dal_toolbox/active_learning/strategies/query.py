from abc import ABC, abstractmethod

from ..data import ActiveLearningDataModule
from ...utils import setup_rng
from ...models.base import BaseModule


class Query(ABC):
    """
    Abstract base class for active learning query strategies.

    Subclasses must implement the `query` method to define how new data points 
    are selected for annotation during an active learning cycle.

    Attributes:
        random_seed (int, optional): Seed for reproducible randomness.
        rng (np.random.Generator): Random number generator initialized with the given seed.
    """
    def __init__(self, random_seed=None):
        """
        Initializes the Query strategy.

        Args:
            random_seed (int, optional): Random seed for reproducibility. If None, RNG will be non-deterministic.
        """
        self.random_seed = random_seed

        # set rng which should be used for all random stuff
        self.rng = setup_rng(random_seed)

    @abstractmethod
    def query(self, *, model: BaseModule, al_datamodule: ActiveLearningDataModule, acq_size: int):
        """
        Abstract method to select data points to acquire (i.e., label).

        Args:
            model (BaseModule): The model currently being trained.
            al_datamodule (ActiveLearningDataModule): The data module that manages labeled/unlabeled splits.
            acq_size (int): Number of samples to acquire in this iteration.

        Returns:
            indices (np.ndarray or list): Indices of selected unlabeled samples to be labeled.
        """
        pass
