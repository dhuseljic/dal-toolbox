import warnings

import torchvision
from enum import Enum
from .base import BaseData, BaseTransforms


class SVHNTransforms(Enum):
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.247, 0.243, 0.262)


class SVHNStandardTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.mean = SVHNTransforms.mean.value
        self.std = SVHNTransforms.std.value

    @property
    def train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def eval_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def query_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform


class SVHN(BaseData):
    def __init__(
            self,
            dataset_path: str,
            transforms: BaseTransforms = None,
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        transforms = SVHNStandardTransforms() if transforms is None else transforms
        self.train_transform = transforms.train_transform
        self.eval_transform = transforms.eval_transform
        self.query_transform = transforms.query_transform
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


def build_svhn(split, ds_path, mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970), return_info=False):
    warnings.warn('Deprecated method build_svhn.')
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    if split == 'train':
        ds = torchvision.datasets.SVHN(ds_path, split='train', download=True, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.SVHN(ds_path, split='train', download=True, transform=eval_transform)
    elif split == 'test':
        ds = torchvision.datasets.SVHN(ds_path, split='test', download=True, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
