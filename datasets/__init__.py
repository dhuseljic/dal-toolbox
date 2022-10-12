import torch
import torchvision
import numpy as np

from torch.utils.data import Subset
from torchvision import transforms
from datasets.presets import ClassificationPresetTrain, ClassificationPresetEval
from datasets.tinyImageNet import TinyImageNet

from .corruptions import GaussianNoise


def build_mnist(split):
    mean, std = (0.1307,), (0.3081,)
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.MNIST('data/', train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform)
    return ds

def build_fashionmnist(split):
    #TODO: I took the same transformations from MNIST since i couldn't find any better
    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform)
    return ds


def build_cifar10(split):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=eval_transform)
    return ds

def build_cifar10_c(severity):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            GaussianNoise(severity)
        ])
    ds = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=eval_transform)
    return ds


def build_cifar100(split):
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    if split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=eval_transform)
    return ds


def build_svhn(split):
    mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    if split == 'train':
        train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std, hflip_prob=0.5)
        ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=train_transform)
    elif split == 'test':
        eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
        ds = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=eval_transform)
    return ds

def build_imagenet(split, path):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if split == 'train':
        train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std)
        ds = torchvision.datasets.ImageNet(root=path, split=split, transform=train_transform)
    elif split == 'val':
        eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
        ds = torchvision.datasets.ImageNet(root=path, split=split, transform=eval_transform)
    return ds

def build_tinyimagenet(split):
    #TODO: Check if mean, std is correct
    mean, std = (120.0378, 111.3496, 106.5628), (73.6951, 69.0155, 69.3879)
    if split == 'train':
        train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std)
        ds = TinyImageNet(root="./data", split='train', transform=train_transform)
        #print("Calculating mean and std...")
        #print("Mean: ",torch.mean(torch.tensor(np.array(ds.images)).float(), dim=(0,1,2)))
        #print("Std:",torch.std(torch.tensor(np.array(ds.images)).float(), dim=(0,1,2)))
    elif split == 'test':
        eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
        ds = TinyImageNet(root="./data", split='test', transform=eval_transform)
    return ds


def build_dataset(args):
    # At first build the in domain dataset
    if args.dataset == 'MNIST':
        train_ds = build_mnist('train')
        test_ds_id = build_mnist('test')
        n_classes = 10

    elif args.dataset == 'MNIST04':
        train_ds = build_mnist('train')
        test_ds_id = build_mnist('test')
        indices_id = (train_ds.targets < 5).nonzero().flatten()
        train_ds = Subset(train_ds, indices=indices_id)
        indices_id = (test_ds_id.targets < 5).nonzero().flatten()
        test_ds_id = Subset(test_ds_id, indices=indices_id)
        n_classes = 5

    elif args.dataset == 'MNIST59':
        train_ds = build_mnist('train')
        test_ds_id = build_mnist('test')
        indices_id = (train_ds.targets >= 5).nonzero().flatten()
        train_ds = Subset(train_ds, indices=indices_id)
        indices_id = (test_ds_id.targets >= 5).nonzero().flatten()
        test_ds_id = Subset(test_ds_id, indices=indices_id)
        n_classes = 5

    elif args.dataset == 'FashionMNIST':
        train_ds = build_fashionmnist('train')
        test_ds_id = build_fashionmnist('test')
        n_classes = 10

    elif args.dataset == 'CIFAR10':
        train_ds = build_cifar10('train')
        test_ds_id = build_cifar10('test')
        n_classes = 10

    elif args.dataset == 'CIFAR100':
        train_ds = build_cifar100('train')
        test_ds_id = build_cifar100('test')
        n_classes = 100

    elif args.dataset == 'SVHN':
        train_ds = build_svhn('train')
        test_ds_id = build_svhn('test')
        n_classes = 10

    elif args.dataset == 'tinyImagenet':
        train_ds = build_tinyimagenet('train')
        test_ds_id = build_tinyimagenet('test')
        n_classes = 200

    elif args.dataset == 'Imagenet':
        train_ds = build_imagenet('train', args.imagenet_path)
        test_ds_id = build_imagenet('val', args.imagenet_path)
        n_classes = 1000

    else:
        raise NotImplementedError

    # Second, choose out of domain datasets
    #TODO: Own methods for each dataset-type
    test_dss_ood = {}
    if 'MNIST' in args.ood_datasets:
        temp_ds = build_mnist('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["MNIST"] = temp_ds

    if 'MNIST04' in args.ood_datasets:
        temp_ds = build_mnist('test')
        indices_ood = (temp_ds.targets < 5).nonzero().flatten()
        temp_ds = Subset(temp_ds, indices=indices_ood)
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["MNIST04"] = temp_ds

    if 'MNIST59' in args.ood_datasets:
        temp_ds = build_mnist('test')
        indices_ood = (temp_ds.targets >= 5).nonzero().flatten()
        temp_ds = Subset(temp_ds, indices=indices_ood)
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["MNIST59"] = temp_ds

    if 'FashionMNIST' in args.ood_datasets:
        temp_ds = build_fashionmnist('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["FashionMNIST"] = temp_ds
    
    if 'CIFAR10' in args.ood_datasets:
        temp_ds = build_cifar10('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["CIFAR10"] = temp_ds

    if 'CIFAR10-C' in args.ood_datasets:
        temp_ds = build_cifar10_c(.5)
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["CIFAR10-C"] = temp_ds

    if 'CIFAR100' in args.ood_datasets:
        temp_ds = build_cifar100('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["CIFAR100"] = temp_ds

    if 'SVHN' in args.ood_datasets:
        temp_ds = build_svhn('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["SVHN"] = temp_ds

    if 'tinyImagenet' in args.ood_datasets:
        temp_ds = build_tinyimagenet('test')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["tinyImagenet"] = temp_ds

    if 'Imagenet' in args.ood_datasets:
        temp_ds = build_imagenet('val')
        test_ds_id, temp_ds = equal_set_sizes(test_ds_id, temp_ds)
        test_dss_ood["Imagenet"] = temp_ds

    # Reduce trainset size if wished
    if args.n_samples:
        indices_id = torch.randperm(len(train_ds))[:args.n_samples]
        train_ds = Subset(train_ds, indices=indices_id)

    assert len(test_dss_ood) != 0, "No ood dataset has been chosen!"
    assert [len(test_ds_id) == len(t_ds_ood) for t_ds_ood in test_dss_ood.values()].count(0) == 0, 'All test sets should have the same size!'
    return train_ds, test_ds_id, test_dss_ood, n_classes


def equal_set_sizes(ds_id, ds_ood):
    # Make test id and ood the same size
    n_samples_id = len(ds_id)
    n_samples_ood = len(ds_ood)
    if n_samples_id != n_samples_ood:
        if n_samples_id < n_samples_ood:
            rnd_indices = torch.randperm(n_samples_ood)[:n_samples_id]
            ds_ood = Subset(ds_ood, indices=rnd_indices)
        else:
            rnd_indices = torch.randperm(n_samples_id)[:n_samples_ood]
            ds_id = Subset(ds_id, indices=rnd_indices)
    return ds_id, ds_ood