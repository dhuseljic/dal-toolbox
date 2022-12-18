import torch
import torchvision
import numpy as np

from torch.utils.data import Subset
from . import mnist, fashion_mnist, svhn, cifar, tiny_imagenet, imagenet


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


def build_ssl_dataset(args):
    if args.dataset == 'CIFAR10':
        train_ds, ds_info = cifar.build_cifar10('ssl_weak', args.dataset_path, return_info=True)
        query_ds = cifar.build_cifar10('ssl_weak', args.dataset_path)
        test_ds_id = cifar.build_cifar10('test', args.dataset_path)

        lb_idx, ulb_idx = sample_labeled_unlabeled_data(
            train_ds.targets, ds_info['n_classes'], lb_num_labels=args.n_labeled_samples,
            ulb_num_labels=args.n_unlabeled_samples)
    else:
        raise NotImplementedError
    return Subset(train_ds, lb_idx), Subset(query_ds, ulb_idx), test_ds_id, ds_info



def sample_labeled_unlabeled_data(target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # get samples per class
    if lb_imbalance_ratio == 1.0:
        # balanced setting, lb_num_labels is total number of labels for labeled data
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
        lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)


    if ulb_imbalance_ratio == 1.0:
        # balanced setting
        if ulb_num_labels is not None and ulb_num_labels != 'None':
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be divideable by num_classes in balanced setting"
            ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
        else:
            ulb_samples_per_class = None
    else:
        # imbalanced setting
        assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
        ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.array([i for i in range(len(target)) if target[i] == c])
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_samples_per_class is None:
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    return lb_idx, ulb_idx


def make_imbalance_data(max_num_labels, num_classes, gamma):
    mu = np.power(1 / abs(gamma), 1 / (num_classes - 1))
    samples_per_class = []
    for c in range(num_classes):
        if c == (num_classes - 1):
            samples_per_class.append(int(max_num_labels / abs(gamma)))
        else:
            samples_per_class.append(int(max_num_labels * np.power(mu, c)))
    if gamma < 0:
        samples_per_class = samples_per_class[::-1]
    return samples_per_class
