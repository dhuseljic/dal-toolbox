import warnings
from enum import Enum

import torchvision
from torchvision import transforms, datasets

from .base import AbstractData
from .corruptions import GaussianNoise
from .utils import ContrastiveTransformations


class CIFAR10Transforms(Enum):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.262)


class CIFAR10(AbstractData):

    def __init__(
            self,
            dataset_path: str,
            mean: tuple = CIFAR10Transforms.mean.value,
            std: tuple = CIFAR10Transforms.std.value,
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        self.mean = mean
        self.std = std
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        datasets.CIFAR10(self.dataset_path, train=True, download=True)
        datasets.CIFAR10(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return datasets.CIFAR10(self.dataset_path, train=True, transform=self.train_transforms)

    @property
    def full_train_dataset_eval_transforms(self):
        return datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transforms)

    @property
    def full_train_dataset_query_transforms(self):
        return datasets.CIFAR10(self.dataset_path, train=True, transform=self.query_transforms)

    @property
    def test_dataset(self):
        return datasets.CIFAR10(self.dataset_path, train=False, transform=self.eval_transforms)

    @property
    def train_transforms(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def eval_transforms(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform

    @property
    def query_transforms(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform


class CIFAR10C(CIFAR10):
    def __init__(
            self,
            dataset_path: str,
            severity: float,
            mean: tuple = CIFAR10Transforms.mean.value,
            std: tuple = CIFAR10Transforms.std.value,
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        self.severity = severity
        assert 0. <= self.severity <= 1.
        super().__init__(dataset_path, mean, std, val_split, seed)

    @property
    def eval_transforms(self):
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            GaussianNoise(self.severity)
        ])
        return eval_transform


class CIFAR10Plain(CIFAR10):
    """
    Version of CIFAR10 which hast only ``transforms.ToTensor()`` as transform.
    """
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None) -> None:
        super().__init__(dataset_path, val_split=val_split, seed=seed)

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.ToTensor(), ])

    @property
    def query_transforms(self):
        return transforms.Compose([transforms.ToTensor(), ])

    @property
    def eval_transforms(self):
        return transforms.Compose([transforms.ToTensor(), ])


class CIFAR10Contrastive(CIFAR10):
    """
    Contrastive version of CIFAR10.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """
    def __init__(self,
                 dataset_path: str,
                 mean: tuple = CIFAR10Transforms.mean.value,
                 std: tuple = CIFAR10Transforms.std.value,
                 val_split: float = 0.1,
                 seed: int = None,
                 color_distortion_strength: float = 1.0
                 ) -> None:
        """

        Args:
            color_distortion_strength: Strength of color jittering transform.
        """
        self.s = color_distortion_strength
        super().__init__(dataset_path, mean, std, val_split, seed)

    @property
    def train_transforms(self):
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur not used in original paper, however, other papers assign it importance
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std),  # TODO (dhuseljic) Discuss if this should be used
        ])
        return ContrastiveTransformations(transform, n_views=2)

    @property
    def eval_transforms(self):
        return self.train_transforms  # TODO This depends on the error we want to calculate on the validation/test set


class CIFAR100(CIFAR10):
    def __init__(
            self,
            dataset_path: str,
            mean: tuple = (0.5071, 0.4865, 0.4409),
            std: tuple = (0.2673, 0.2564, 0.2762),
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        super().__init__(dataset_path, mean, std, val_split, seed)

    @property
    def num_classes(self):
        return 100

    def download_datasets(self):
        datasets.CIFAR100(self.dataset_path, train=True, download=True)
        datasets.CIFAR100(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return datasets.CIFAR100(self.dataset_path, train=True, transform=self.train_transforms)

    @property
    def full_train_dataset_eval_transforms(self):
        return datasets.CIFAR100(self.dataset_path, train=True, transform=self.eval_transforms)

    @property
    def full_train_dataset_query_transforms(self):
        return datasets.CIFAR100(self.dataset_path, train=True, transform=self.query_transforms)

    @property
    def test_dataset(self):
        return datasets.CIFAR100(self.dataset_path, train=False, transform=self.eval_transforms)


def build_cifar10(split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262), return_info=False):
    warnings.warn('Deprecated method build_cifar10.')
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
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
