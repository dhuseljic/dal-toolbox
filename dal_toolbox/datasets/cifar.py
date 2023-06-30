import warnings
from enum import Enum

import torchvision

from .base import BaseData, BaseTransforms
from .corruptions import GaussianNoise
from .utils import ContrastiveTransformations


class PlainTransforms(BaseTransforms):
    @property
    def train_transform(self):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    @property
    def query_transform(self):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    @property
    def eval_transform(self):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])


class CIFAR10Transforms(Enum):
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.247, 0.243, 0.262)


class CIFAR10StandardTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.mean = CIFAR10Transforms.mean.value
        self.std = CIFAR10Transforms.std.value

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


class CIFAR10(BaseData):

    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        transforms = CIFAR10StandardTransforms() if transforms is None else transforms
        self.train_transform = transforms.train_transform
        self.eval_transform = transforms.eval_transform
        self.query_transform = transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        torchvision.datasets.CIFAR10(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR10(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=False, transform=self.eval_transform)


class CIFAR10Plain(CIFAR10):
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None) -> None:
        super().__init__(dataset_path, PlainTransforms(), val_split, seed)


class CIFAR10CTransforms(CIFAR10StandardTransforms):
    def __init__(self, severity: float):
        super().__init__()
        self.severity = severity

    @property
    def eval_transform(self):
        eval_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            GaussianNoise(self.severity)
        ])
        return eval_transform


class CIFAR10C(CIFAR10):
    def __init__(self,
                 dataset_path: str,
                 severity: float = .5,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        super().__init__(dataset_path, CIFAR10CTransforms(severity=severity), val_split, seed)


class CIFAR10ContrastiveTransforms(CIFAR10StandardTransforms):
    def __init__(self, color_distortion_strength: float = 1.0):
        super().__init__()
        self._s = color_distortion_strength

    @property
    def train_transform(self):
        color_jitter = torchvision.transforms.ColorJitter(0.8 * self._s, 0.8 * self._s, 0.8 * self._s, 0.2 * self._s)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=32),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            # GaussianBlur not used in original paper, however, other papers assign it importance
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=3)], p=0.5),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std),  # TODO (dhuseljic) Discuss if this should be used
        ])
        return ContrastiveTransformations(transform)

    @property
    def eval_transform(self):
        return self.train_transform


class CIFAR10Contrastive(CIFAR10):
    """
    Contrastive version of CIFAR10.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """

    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None) -> None:
        super().__init__(dataset_path, CIFAR10ContrastiveTransforms(), val_split, seed)


class CIFAR100Transforms(Enum):
    mean: tuple = (0.5071, 0.4865, 0.4409)
    std: tuple = (0.2673, 0.2564, 0.2762)


class CIFAR100StandardTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.mean = CIFAR100Transforms.mean.value
        self.std = CIFAR100Transforms.std.value

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


class CIFAR100(BaseData):

    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        transforms = CIFAR100StandardTransforms() if transforms is None else transforms
        self.train_transform = transforms.train_transform
        self.eval_transform = transforms.eval_transform
        self.query_transform = transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 100

    def download_datasets(self):
        torchvision.datasets.CIFAR100(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR100(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=False, transform=self.eval_transform)


def build_cifar10(split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262), return_info=False):
    warnings.warn('Deprecated method build_cifar10.')
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    if split == 'train':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=eval_transform)
    elif split == 'raw':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True)
    elif split == 'test':
        ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True, transform=eval_transform)

    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds


def build_cifar100(split, ds_path, mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762), return_info=False):
    warnings.warn('Deprecated method build_cifar100.')
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    if split == 'train':
        ds = torchvision.datasets.CIFAR100(ds_path, train=True, download=True, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.CIFAR100(ds_path, train=True, download=True, transform=eval_transform)
    elif split == 'test':
        ds = torchvision.datasets.CIFAR100(ds_path, train=False, download=True, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 100, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
