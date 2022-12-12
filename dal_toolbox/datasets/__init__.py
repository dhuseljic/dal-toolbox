import torch
import torchvision
from datasets import load_dataset
import numpy as np

from torch.utils.data import Subset
from . import mnist, fashion_mnist, svhn, cifar, tiny_imagenet, imagenet, imdb


def build_dataset(args):
    if args.dataset == 'MNIST':
        train_ds, ds_info = mnist.build_mnist('train', args.dataset_path, return_info=True)
        test_ds = mnist.build_mnist('test', args.dataset_path)

    elif args.dataset == 'FashionMNIST':
        train_ds, ds_info = fashion_mnist.build_fashionmnist('train', args.dataset_path, return_info=True)
        test_ds = fashion_mnist.build_fashionmnist('test', args.dataset_path)

    elif args.dataset == 'CIFAR10':
        train_ds, ds_info = cifar.build_cifar10('train', args.dataset_path, return_info=True)
        test_ds = cifar.build_cifar10('test', args.dataset_path)

    elif args.dataset == 'CIFAR100':
        train_ds, ds_info = cifar.build_cifar100('train', args.dataset_path, return_info=True)
        test_ds = cifar.build_cifar100('test', args.dataset_path)

    elif args.dataset == 'SVHN':
        train_ds, ds_info = svhn.build_svhn('train', args.dataset_path, return_info=True)
        test_ds = svhn.build_svhn('test', args.dataset_path)

    elif args.dataset == 'TinyImagenet':
        train_ds, ds_info = tiny_imagenet.build_tinyimagenet('train', args.dataset_path, return_info=True)
        test_ds = tiny_imagenet.build_tinyimagenet('test', args.dataset_path)

    elif args.dataset == 'Imagenet':
        train_ds, ds_info = imagenet.build_imagenet('train', args.dataset_path)
        test_ds = imagenet.build_imagenet('val', args.dataset_path)
    else:
        raise NotImplementedError

    return train_ds, test_ds, ds_info


def build_ood_datasets(args, mean, std):
    ood_datasets = {}
    if 'MNIST' in args.ood_datasets:
        test_ds_ood = mnist.build_mnist('test', args.dataset_path, mean, std)
        ood_datasets["MNIST"] = test_ds_ood

    if 'FashionMNIST' in args.ood_datasets:
        test_ds_ood = fashion_mnist.build_fashionmnist('test', args.dataset_path, mean, std)
        ood_datasets["FashionMNIST"] = test_ds_ood

    if 'CIFAR10' in args.ood_datasets:
        test_ds_ood = cifar.build_cifar10('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR10"] = test_ds_ood

    if 'CIFAR10-C' in args.ood_datasets:
        test_ds_ood = cifar.build_cifar10_c(.5, args.dataset_path, mean, std)
        ood_datasets["CIFAR10-C"] = test_ds_ood

    if 'CIFAR100' in args.ood_datasets:
        test_ds_ood = cifar.build_cifar100('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR100"] = test_ds_ood

    if 'SVHN' in args.ood_datasets:
        test_ds_ood = svhn.build_svhn('test', args.dataset_path, mean, std)
        ood_datasets["SVHN"] = test_ds_ood

    if 'TinyImagenet' in args.ood_datasets:
        test_ds_ood = tiny_imagenet.build_tinyimagenet('test', args.dataset_path, mean, std)
        ood_datasets["TinyImagenet"] = test_ds_ood

    if 'Imagenet' in args.ood_datasets:
        test_ds_ood = imagenet.build_imagenet('val', args.dataset_path, mean, std)
        ood_datasets["Imagenet"] = test_ds_ood

    return ood_datasets


def build_al_datasets(args):
    if args.dataset == 'MNIST':
        train_ds, ds_info = mnist.build_mnist('train', args.dataset_path, return_info=True)
        query_ds = mnist.build_mnist('query', args.dataset_path)
        test_ds_id = mnist.build_mnist('test', args.dataset_path)

    elif args.dataset == 'FashionMNIST':
        train_ds, ds_info = fashion_mnist.build_fashionmnist('train', args.dataset_path, return_info=True)
        query_ds = fashion_mnist.build_fashionmnist('query', args.dataset_path)
        test_ds_id = fashion_mnist.build_fashionmnist('test', args.dataset_path)

    elif args.dataset == 'CIFAR10':
        train_ds, ds_info = cifar.build_cifar10('train', args.dataset_path, return_info=True)
        query_ds = cifar.build_cifar10('query', args.dataset_path)
        test_ds_id = cifar.build_cifar10('test', args.dataset_path)

    elif args.dataset == 'CIFAR100':
        train_ds, ds_info = cifar.build_cifar100('train', args.dataset_path, return_info=True)
        query_ds = cifar.build_cifar100('query', args.dataset_path)
        test_ds_id = cifar.build_cifar100('test', args.dataset_path)

    elif args.dataset == 'SVHN':
        train_ds, ds_info = svhn.build_svhn('train', args.dataset_path, return_info=True)
        query_ds = svhn.build_svhn('query', args.dataset_path)
        test_ds_id = svhn.build_svhn('test', args.dataset_path)

    elif args.dataset == 'TinyImagenet':
        train_ds, ds_info = tiny_imagenet.build_tinyimagenet('train', args.dataset_path, return_info=True)
        query_ds = tiny_imagenet.build_tinyimagenet('query', args.dataset_path)
        test_ds_id = tiny_imagenet.build_tinyimagenet('test', args.dataset_path)

    elif args.dataset == 'Imagenet':
        train_ds, ds_info = imagenet.build_imagenet('train', args.dataset_path, return_info=True)
        query_ds = imagenet.build_imagenet('query', args.dataset_path)
        test_ds_id = imagenet.build_imagenet('val', args.dataset_path)
    
    elif args.dataset == 'imdb':
        raw_ds = load_dataset('imdb')
        train_ds, ds_info = imdb.build_imdb('train', raw_ds, args)
        query_ds = imdb.build_imdb('query', raw_ds, args)
        test_ds_id = imdb.build_imdb('test', raw_ds, args)
    else:
        raise NotImplementedError

    return train_ds, query_ds, test_ds_id, ds_info


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
