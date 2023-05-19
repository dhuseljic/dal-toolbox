import warnings
import torchvision
import lightning as L

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from .corruptions import GaussianNoise
from .base import AbstractData


class CIFAR10(AbstractData):
    @property
    def mean(self):
        return (0.4914, 0.4822, 0.4465)

    @property
    def std(self):
        return (0.247, 0.243, 0.262)

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
    def __init__(self, dataset_path: str, severity: float, val_split: float = 0.1, seed: int = None) -> None:
        self.severity = severity
        super().__init__(dataset_path, val_split, seed)

    @property
    def eval_transforms(self):
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            GaussianNoise(self.severity)
        ])
        return eval_transform


class CIFAR100(CIFAR10):
    @property
    def mean(self):
        return (0.5071, 0.4865, 0.4409)

    @property
    def std(self):
        return (0.2673, 0.2564, 0.2762)

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
