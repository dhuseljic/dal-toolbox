import torchvision
from torchvision import transforms
from .corruptions import GaussianNoise, RandAugment


def build_cifar10(split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262), return_info=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ssl_transform_weak = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ssl_transform_strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == 'train':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=eval_transform)
    elif split == 'ssl_weak':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=ssl_transform_weak)
    elif split == 'ssl_strong':
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=ssl_transform_strong)
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



