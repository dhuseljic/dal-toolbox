from enum import Enum

import torchvision

from .base import BaseData, BaseTransforms
from .utils import PlainTransforms


class Country211Transforms(Enum):
    # TODO
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class Country211(BaseData):

    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        self.transforms = PlainTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 211

    def download_datasets(self):
        torchvision.datasets.Country211(self.dataset_path, split="train", download=True)
        torchvision.datasets.Country211(self.dataset_path, split="test", download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.Country211(self.dataset_path, split="train", transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.Country211(self.dataset_path, split="train", transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.Country211(self.dataset_path, split="train", transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.Country211(self.dataset_path, split="test", transform=self.eval_transform)