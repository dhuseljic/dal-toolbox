import warnings

import torchvision
from enum import Enum
from .base import BaseData, BaseTransforms
from .transforms import StandardTransforms

SVHN_MEAN = (0.4914, 0.4822, 0.4465)
SVHN_STD = (0.247, 0.243, 0.262)


class SVHNStandardTransforms(StandardTransforms):
    def __init__(self):
        super().__init__(resize=32, mean=SVHN_MEAN, std=SVHN_STD)


class SVHN(BaseData):
    def __init__(
            self,
            dataset_path: str,
            transforms: BaseTransforms = None,
            val_split: float = 0.1,
            seed: int = None) -> None:
        self.transforms = SVHNStandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        torchvision.datasets.SVHN(self.dataset_path, split='train', download=True)
        torchvision.datasets.SVHN(self.dataset_path, split='test', download=True)
        # torchvision.datasets.SVHN(self.dataset_path, split='extra', download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='test', transform=self.eval_transform)
