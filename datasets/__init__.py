import torch
import torchvision
import numpy as np

from torch.utils.data import Subset
from torchvision import transforms
from datasets.presets import ClassificationPresetTrain, ClassificationPresetEval
from datasets.tinyImageNet import TinyImageNet

from .corruptions import GaussianNoise


def build_mnist(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.1307,), (0.3081,)
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.MNIST(ds_path, train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.MNIST(ds_path, train=False, download=True, transform=transform)
    return ds


def build_fashionmnist(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.5,), (0.5,)
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.FashionMNIST(ds_path, train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.FashionMNIST(ds_path, train=False, download=True, transform=transform)
    return ds


def build_cifar10(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True, transform=eval_transform)
    return ds


def build_cifar100(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    if split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR100(ds_path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR100(ds_path, train=False, download=True, transform=eval_transform)
    return ds


def build_svhn(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    if split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.SVHN(ds_path, split='train', download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = None
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = torchvision.datasets.SVHN(ds_path, split='test', download=True, transform=eval_transform)
    return ds


def build_imagenet(split, ds_path, mean=None, std=None):
    if not mean:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if split == 'train':
        train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std)
        ds = torchvision.datasets.ImageNet(root=ds_path, split=split, transform=train_transform)
    elif split == 'val':
        eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
        ds = torchvision.datasets.ImageNet(root=ds_path, split=split, transform=eval_transform)
    return ds


def build_tinyimagenet(split, ds_path, mean=None, std=None):
    if not mean:
        # TODO: Check if mean, std is correct
        mean, std = (120.0378, 111.3496, 106.5628), (73.6951, 69.0155, 69.3879)
    if split == 'train':
        train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std)
        ds = TinyImageNet(root=ds_path, split='train', transform=train_transform)
        #print("Calculating mean and std...")
        #print("Mean: ",torch.mean(torch.tensor(np.array(ds.images)).float(), dim=(0,1,2)))
        #print("Std:",torch.std(torch.tensor(np.array(ds.images)).float(), dim=(0,1,2)))
    elif split == 'test':
        eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
        ds = TinyImageNet(root=ds_path, split='test', transform=eval_transform)
    return ds


def build_cifar10_c(severity, ds_path, mean=None, std=None):
    # TODO: Set Mean to mean and not to none, same with std
    if not mean:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        GaussianNoise(severity)
    ])
    ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True, transform=eval_transform)
    return ds


def build_dataset(args):
    ds_info = {}
    if args.dataset == 'MNIST':
        ds_info['n_classes'] = 10
        ds_info['mean'] = (0.1307,)
        ds_info['std'] = (0.3081,)
        train_ds = build_mnist('train', args.dataset_path)
        test_ds = build_mnist('test', args.dataset_path)

    elif args.dataset == 'FashionMNIST':
        ds_info['n_classes'] = 10
        ds_info['mean'] = (0.5,)
        ds_info['std'] = (0.5,)
        train_ds = build_fashionmnist('train', args.dataset_path)
        test_ds = build_fashionmnist('test', args.dataset_path)

    elif args.dataset == 'CIFAR10':
        ds_info['n_classes'] = 10
        ds_info['mean'] = (0.4914, 0.4822, 0.4465)
        ds_info['std'] = (0.2023, 0.1994, 0.2010)
        train_ds = build_cifar10('train', args.dataset_path)
        test_ds = build_cifar10('test', args.dataset_path)

    elif args.dataset == 'CIFAR100':
        ds_info['n_classes'] = 100
        ds_info['mean'] = (0.5071, 0.4865, 0.4409)
        ds_info['std'] = (0.2673, 0.2564, 0.2762)
        train_ds = build_cifar100('train', args.dataset_path)
        test_ds = build_cifar100('test', args.dataset_path)

    elif args.dataset == 'SVHN':
        ds_info['n_classes'] = 10
        ds_info['mean'] = (0.4377, 0.4438, 0.4728)
        ds_info['std'] = (0.1980, 0.2010, 0.1970)
        train_ds = build_svhn('train', args.dataset_path)
        test_ds = build_svhn('test', args.dataset_path)

    elif args.dataset == 'TinyImagenet':
        # TODO: Check mean and std
        mean, std = (120.0378, 111.3496, 106.5628), (73.6951, 69.0155, 69.3879)
        train_ds = build_tinyimagenet('train', args.dataset_path)
        test_ds = build_tinyimagenet('test', args.dataset_path)
        n_classes = 200

    elif args.dataset == 'Imagenet':
        ds_info['n_classes'] = 1000
        ds_info['mean'] = (0.485, 0.456, 0.406)
        ds_info['std'] = (0.229, 0.224, 0.225)
        train_ds = build_imagenet('train', args.dataset_path)
        test_ds = build_imagenet('val', args.dataset_path)
    else:
        raise NotImplementedError

    return train_ds, test_ds, ds_info


def build_ood_datasets(args, mean, std):
    ood_datasets = {}
    if 'MNIST' in args.ood_datasets:
        test_ds_ood = build_mnist('test', args.dataset_path, mean, std)
        ood_datasets["MNIST"] = test_ds_ood

    if 'FashionMNIST' in args.ood_datasets:
        test_ds_ood = build_fashionmnist('test', args.dataset_path, mean, std)
        ood_datasets["FashionMNIST"] = test_ds_ood

    if 'CIFAR10' in args.ood_datasets:
        test_ds_ood = build_cifar10('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR10"] = test_ds_ood

    if 'CIFAR10-C' in args.ood_datasets:
        test_ds_ood = build_cifar10_c(.5, args.dataset_path, mean, std)
        ood_datasets["CIFAR10-C"] = test_ds_ood

    if 'CIFAR100' in args.ood_datasets:
        test_ds_ood = build_cifar100('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR100"] = test_ds_ood

    if 'SVHN' in args.ood_datasets:
        test_ds_ood = build_svhn('test', args.dataset_path, mean, std)
        ood_datasets["SVHN"] = test_ds_ood

    if 'TinyImagenet' in args.ood_datasets:
        test_ds_ood = build_tinyimagenet('test', args.dataset_path, mean, std)
        ood_datasets["TinyImagenet"] = test_ds_ood

    if 'Imagenet' in args.ood_datasets:
        test_ds_ood = build_imagenet('val', args.dataset_path, mean, std)
        ood_datasets["Imagenet"] = test_ds_ood

    return ood_datasets


def build_al_datasets(args):
    if args.dataset == 'MNIST':
        train_ds = build_mnist('train', args.dataset_path)
        test_ds_id = build_mnist('test', args.dataset_path)
        n_classes = 10

    elif args.dataset == 'FashionMNIST':
        train_ds = build_fashionmnist('train', args.dataset_path)
        test_ds_id = build_fashionmnist('test', args.dataset_path)
        n_classes = 10

    elif args.dataset == 'CIFAR10':
        train_ds = build_cifar10('train', args.dataset_path)
        test_ds_id = build_cifar10('test', args.dataset_path)
        n_classes = 10

    elif args.dataset == 'CIFAR100':
        train_ds = build_cifar100('train', args.dataset_path)
        test_ds_id = build_cifar100('test', args.dataset_path)
        n_classes = 100

    elif args.dataset == 'SVHN':
        train_ds = build_svhn('train', args.dataset_path)
        test_ds_id = build_svhn('test', args.dataset_path)
        n_classes = 10

    elif args.dataset == 'TinyImagenet':
        train_ds = build_tinyimagenet('train', args.dataset_path)
        test_ds_id = build_tinyimagenet('test', args.dataset_path)
        n_classes = 200

    elif args.dataset == 'Imagenet':
        train_ds = build_imagenet('train', args.dataset_path)
        test_ds_id = build_imagenet('val', args.dataset_path)
        n_classes = 1000

    else:
        raise NotImplementedError

    # TODO: add query_ds
    query_ds = train_ds

    return train_ds, query_ds, test_ds_id, n_classes


def equal_set_sizes(ds_id, ds_ood):
    # Make test id and ood the same size
    n_samples_id = len(ds_id)
    n_samples_ood = len(ds_ood)
    if n_samples_id < n_samples_ood:
        rnd_indices = torch.randperm(n_samples_ood)[:n_samples_id]
        ds_ood = Subset(ds_ood, indices=rnd_indices)
    elif n_samples_id > n_samples_ood:
        rnd_indices = torch.randperm(n_samples_id)[:n_samples_ood]
        ds_id = Subset(ds_id, indices=rnd_indices)
    return ds_id, ds_ood
