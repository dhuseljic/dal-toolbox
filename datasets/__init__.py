import torch
import torchvision

from torch.utils.data import Subset
from torchvision import transforms
from datasets.presets import ClassificationPresetTrain, ClassificationPresetEval


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


def build_dataset(args):
    if args.dataset == 'MNIST_vs_MNIST':
        train_ds = build_mnist('train')
        test_ds = build_mnist('test')

        # Prepare train -> mnist 0 to 4 vs 5 to 9
        indices_id = (train_ds.targets < 5).nonzero().flatten()
        train_ds = Subset(train_ds, indices=indices_id)

        indices_id = (test_ds.targets < 5).nonzero().flatten()
        test_ds_id = Subset(test_ds, indices=indices_id)
        indices_ood = (test_ds.targets >= 5).nonzero().flatten()
        test_ds_ood = Subset(test_ds, indices=indices_ood)
        n_classes = 5
    elif args.dataset == 'CIFAR10_vs_CIFAR100':
        train_ds = build_cifar10('train')
        test_ds_id = build_cifar10('test')
        test_ds_ood = build_cifar100('test')
        n_classes = 10
    elif args.dataset == 'CIFAR10_vs_SVHN':
        train_ds = build_cifar10('train')
        test_ds_id = build_cifar10('test')
        test_ds_ood = build_svhn('test')
        n_classes = 10
    elif args.dataset == 'CIFAR100_vs_CIFAR10':
        train_ds = build_cifar100('train')
        test_ds_id = build_cifar100('test')
        test_ds_ood = build_cifar10('test')
        n_classes = 100
    elif args.dataset == 'CIFAR100_vs_SVHN':
        train_ds = build_cifar100('train')
        test_ds_id = build_cifar100('test')
        test_ds_ood = build_svhn('test')
        n_classes = 100
    else:
        raise NotImplementedError

    # make test id and ood the same size
    n_samples_id = len(test_ds_id)
    n_samples_ood = len(test_ds_ood)
    if n_samples_id != n_samples_ood:
        if n_samples_id < n_samples_ood:
            rnd_indices = torch.randperm(n_samples_ood)[:n_samples_id]
            test_ds_ood = Subset(test_ds_ood, indices=rnd_indices)
        else:
            rnd_indices = torch.randperm(n_samples_id)[:n_samples_ood]
            test_ds_id = Subset(test_ds_id, indices=rnd_indices)

    if args.n_samples:
        indices_id = torch.randperm(len(train_ds))[:args.n_samples]
        train_ds = Subset(train_ds, indices=indices_id)

    assert len(test_ds_id) == len(test_ds_ood)
    return train_ds, test_ds_id, test_ds_ood, n_classes
