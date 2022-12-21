import torch
from torch.utils.data import Subset
from . import mnist, fashion_mnist, svhn, cifar, tiny_imagenet, imagenet, imdb
from .activeglae import agnews, banks77, dbpedia, fnc1, mnli, qnli, sst2, trec6, wikitalk, yelp5

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

    #TODO: cache dir allgemein funktionsfähig
    elif args.dataset.name == 'imdb':
        ds, ds_info = imdb.build_imdb(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'agnews':
        ds, ds_info = agnews.build_agnews(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']
    
    elif args.dataset.name == 'banks77':
        ds, ds_info = banks77.build_banks77(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'dbpedia':
        ds, ds_info = dbpedia.build_dbpedia(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']
    
    elif args.dataset.name == 'fnc1':
        ds, ds_info = fnc1.build_fnc1(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'mnli':
        ds, ds_info = mnli.build_mnli(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'qnli':
        ds, ds_info = qnli.build_qnli(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'sst2':
        ds, ds_info = sst2.build_sst2(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']
    
    elif args.dataset.name == 'trec6':
        ds, ds_info = trec6.build_trec6(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'wikitalk':
        ds, ds_info = wikitalk.build_wikitalk(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    elif args.dataset.name == 'yelp5':
        ds, ds_info = yelp5.build_yelp5(args)
        train_ds = ds['train']
        query_ds = ds['train']
        test_ds_id = ds['test']

    else:
        raise NotImplementedError('Dataset not available')

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

