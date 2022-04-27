from comet_ml import Experiment # fmt: off

import argparse

import torch
import torch.nn as nn

import torchvision

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale

from models import ddu, vanilla, sngp
from models.resnet_spectral_norm import resnet18
from utils import plot_grids


def main(args):
    experiment = Experiment(
        api_key="EzondnlNsOX3ImCKeHrXwAhnG",
        project_name="spectral-norm",
        workspace="huseljic",
    )
    experiment.log_parameters(vars(args))

    print(args)
    train_ds, test_ds_id, test_ds_ood = build_dataset(args)
    fig = plot_grids(train_ds, test_ds_id, test_ds_ood)
    experiment.log_figure(fig)

    train_loader = DataLoader(train_ds, batch_size=64)
    test_loader_id = DataLoader(test_ds_id, batch_size=64*4)
    test_loader_ood = DataLoader(test_ds_ood, batch_size=64*4)

    model_dict = build_model(args, train_ds.n_classes)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']

    history_train, history_test = [], []
    for i_epoch in range(args.n_epochs):
        train_stats = train_one_epoch(model, train_loader, **model_dict['train_params'], epoch=i_epoch)
        history_train.append(train_stats)
        test_stats = evaluate(model, test_loader_id, test_loader_ood, **model_dict['eval_params'])
        history_test.append(test_stats)
        print(f"Epoch [{i_epoch}]", train_stats, test_stats)
        experiment.log_metrics(train_stats, step=i_epoch)
        experiment.log_metrics(test_stats, step=i_epoch)

def build_model(args, n_classes):
    if args.model == 'vanilla':
        model = torchvision.models.resnet18(True)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': vanilla.train_one_epoch,
            'train_params': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'evaluate': vanilla.evaluate,
            'eval_params': dict(criterion=criterion, device=args.device),
        }
    if args.model == 'DDU':
        model = resnet18(
            spectral_normalization=(args.coeff != 0),
            num_classes=n_classes,
            coeff=args.coeff,
        )
        model = ddu.DDUWrapper(model)
        model.n_classes = n_classes
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': ddu.train_one_epoch,
            'train_params': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'evaluate': ddu.evaluate,
            'eval_params': dict(criterion=criterion, device=args.device),
        }
    elif args.model == 'SNGP':
        model = resnet18(
            spectral_normalization=(args.coeff != 0),
            num_classes=n_classes,
            coeff=args.coeff,
        )
        model = sngp.SNGP(model, in_features=512, num_inducing=1024, num_classes=n_classes)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sngp.train_one_epoch,
            'train_params': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'evaluate': sngp.evaluate,
            'eval_params': dict(criterion=criterion, device=args.device),
        }
    else:
        NotImplementedError(f'Model {args.model} not implemented.')

    return model_dict


def build_dataset(args):
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
        train_ds.n_classes = 5
    elif args.dataset == 'CIFAR10_vs_CIFAR100':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        train_ds.n_classes = 10
    elif args.dataset == 'CIFAR10_vs_SVHN':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.SVHN('data/', split='test', download=True, transform=transform)

        # make id and ood the same size
        rnd_indices = torch.randperm(len(test_ds_ood))[:len(test_ds_id)]
        test_ds_ood = Subset(test_ds_ood, indices=rnd_indices)
        train_ds.n_classes = 10
    elif args.dataset == 'CIFAR100_vs_CIFAR10':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)
        train_ds.n_classes = 100
    elif args.dataset == 'CIFAR100_vs_SVHN':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=transform)
        test_ds_id = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=transform)
        test_ds_ood = torchvision.datasets.SVHN('data/', split='test', download=True, transform=transform)

        # make id and ood the same size
        rnd_indices = torch.randperm(len(test_ds_ood))[:len(test_ds_id)]
        test_ds_ood = Subset(test_ds_ood, indices=rnd_indices)
        train_ds.n_classes = 100
    else:
        raise NotImplementedError

    if args.n_samples:
        indices_id = torch.randperm(len(train_ds))[:args.n_samples]
        train_ds_subset = Subset(train_ds, indices=indices_id)
        train_ds_subset.n_classes = train_ds.n_classes
        train_ds = train_ds_subset

    return train_ds, test_ds_id, test_ds_ood


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST_vs_MNIST', choices=[
        'MNIST_vs_MNIST',
        'CIFAR10_vs_CIFAR100',
        'CIFAR10_vs_SVHN',
        'CIFAR100_vs_CIFAR10',
        'CIFAR100_vs_SVHN',
    ])
    parser.add_argument('--model', type=str, default='vanilla', choices=['vanilla', 'DDU', 'SNGP'])
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--coeff', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=.01)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
