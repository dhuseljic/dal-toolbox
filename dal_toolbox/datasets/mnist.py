from enum import Enum

import torchvision
from torchvision import transforms
from .base import BaseData, BaseTransforms


class MNISTTransforms(Enum):
    mean: tuple = (0.1307,)
    std: tuple = (0.3081,)


class MNISTStandardTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.mean = MNISTTransforms.mean.value
        self.std = MNISTTransforms.std.value

    @property
    def train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean*3, self.std*3),
        ])
        return transform

    @property
    def eval_transform(self):
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean*3, self.std*3),
        ])
        return transform

    @property
    def query_transform(self):
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean*3, self.std*3),
        ])
        return transform


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


# TODO(Discuss): Is this outdated?
def build_mnist(split, ds_path, mean=(0.1307,), std=(0.3081,), return_info=False):
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.MNIST(ds_path, train=True, download=True, transform=transform)
    elif split == 'query':
        ds = torchvision.datasets.MNIST(ds_path, train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.MNIST(ds_path, train=False, download=True, transform=transform)
    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
