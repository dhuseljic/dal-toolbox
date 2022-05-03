import torch
import torchvision

from torch.utils.data import Subset
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from datasets.presets import ClassificationPresetTrain, ClassificationPresetEval

def build_dataset(args, seed=42):
    torch.manual_seed(seed)
    if args.dataset == 'MNIST_vs_MNIST':
        # mnist 0 to 4 vs 5 to 9
        transform = Compose([Resize(size=(32, 32)), Grayscale(num_output_channels=3), ToTensor()])
        train_ds = torchvision.datasets.MNIST('data/', train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform)

        # Prepare train
        indices_id = (train_ds.targets < 5).nonzero().flatten()
        train_ds = Subset(train_ds, indices=indices_id)

        indices_id = (test_ds.targets < 5).nonzero().flatten()
        test_ds_id = Subset(test_ds, indices=indices_id)
        indices_ood = (test_ds.targets >= 5).nonzero().flatten()
        test_ds_ood = Subset(test_ds, indices=indices_ood)
        n_classes = 5
    elif args.dataset == 'CIFAR10_vs_CIFAR100':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        n_classes = 10
    elif args.dataset == 'CIFAR10_vs_SVHN':
        # transform = Compose([Resize((32, 32)), ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])])
        train_transform = ClassificationPresetTrain(crop_size=32)
        train_ds = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=train_transform)
        eval_transform = ClassificationPresetEval(crop_size=32)
        test_ds_id = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=eval_transform)
        test_ds_ood = torchvision.datasets.SVHN('data/', split='test', download=True, transform=eval_transform)
        n_classes = 10
    elif args.dataset == 'CIFAR100_vs_CIFAR10':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)
        n_classes = 100
    elif args.dataset == 'CIFAR100_vs_SVHN':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.SVHN('data/', split='test', download=True, transform=transform)

        # make id and ood the same size
        rnd_indices = torch.randperm(len(test_ds_ood))[:len(test_ds_id)]
        test_ds_ood = Subset(test_ds_ood, indices=rnd_indices)
        n_classes = 100
    else:
        raise NotImplementedError
    
    # make id and ood the same size
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

