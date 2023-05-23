import torch
import torchvision
import lightning as L
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from .corruptions import GaussianNoise


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, dataset_path, train_batch_size=64, val_batch_size=64, test_batch_size=64, val_split=.1):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])

    @property
    def mean(self):
        return (0.4914, 0.4822, 0.4465)

    @property
    def std(self):
        return (0.247, 0.243, 0.262)

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR10(self.dataset_path, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            dataset = torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.train_transform)
            dataset_no_aug = torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transform)

            num_samples = len(dataset)
            num_samples_train = int(num_samples * (1 - self.val_split))
            rnd_indices = torch.randperm(num_samples)
            train_indices = rnd_indices[:num_samples_train]
            val_indices = rnd_indices[num_samples_train:]

            self.train_ds = Subset(dataset, indices=train_indices)
            self.val_ds = Subset(dataset_no_aug, indices=val_indices)

        if stage == 'test':
            self.test_ds = torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transform)

        if stage == 'predict':
            self.test_ds = torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.val_batch_size)


def build_cifar10(split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262), return_info=False):
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


def build_cifar10_c(severity, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        GaussianNoise(severity)
    ])
    ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True, transform=eval_transform)
    return ds
