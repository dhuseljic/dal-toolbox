import torch.distributed as dist
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from ..utils import setup_rng
from lightning.pytorch.utilities import rank_zero_warn


class BaseTransforms(ABC):
    @property
    @abstractmethod
    def train_transform(self):
        pass

    @property
    @abstractmethod
    def eval_transform(self):
        pass

    @property
    @abstractmethod
    def query_transform(self):
        pass


class BaseData(ABC):
    def __init__(self, dataset_path: str, val_split: float = .1, seed: int = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.val_split = val_split
        self.rng = setup_rng(seed)

        # Download and get dataset
        self._download_datasets()

        # Train Validation split on training dataset
        train_indices, val_indices = self._get_train_val_indices(len(self.full_train_dataset))

        # Define datasets
        self.train_dataset = Subset(self.full_train_dataset, indices=train_indices)
        self.val_dataset = Subset(self.full_train_dataset_eval_transforms, indices=val_indices)
        self.query_dataset = Subset(self.full_train_dataset_query_transforms, indices=train_indices)

    def _get_train_val_indices(self, num_samples):
        if 0 < self.val_split < 1:
            num_samples_train = int(num_samples * (1 - self.val_split))
        else:
            num_samples_train = num_samples - self.val_split
        rnd_indices = self.rng.permutation(num_samples)
        train_indices = rnd_indices[:num_samples_train]
        val_indices = rnd_indices[num_samples_train:]
        return train_indices, val_indices

    def _download_datasets(self):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                self.download_datasets()
            dist.barrier()  # Make sure that only the process with rank 0 downloads the data
        else:
            self.download_datasets()

    def download_datasets(self):
        rank_zero_warn('Download dataset method not implemented.')

    @property
    @abstractmethod
    def full_train_dataset(self):
        pass

    @property
    @abstractmethod
    def full_train_dataset_eval_transforms(self):
        pass

    @property
    @abstractmethod
    def full_train_dataset_query_transforms(self):
        pass

    @property
    @abstractmethod
    def test_dataset(self):
        pass
