from enum import Enum

import torchvision
from torchvision import transforms
from .base import BaseData, BaseTransforms
from .transforms import CustomTransforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MNISTStandardTransforms(CustomTransforms):
    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN*3, MNIST_STD*3),
        ])
        super().__init__(train_transform=transform, eval_transform=transform)


class MNIST(BaseData):
    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        self.transforms = MNISTStandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        torchvision.datasets.MNIST(self.dataset_path, train=True, download=True)
        torchvision.datasets.MNIST(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.MNIST(self.dataset_path, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.MNIST(self.dataset_path, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.MNIST(self.dataset_path, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.MNIST(self.dataset_path, train=False, transform=self.eval_transform)
