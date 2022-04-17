from comet_ml import Experiment # fmt: off

import argparse

import torch
import torch.nn as nn

import torchvision

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Resize, Compose

from models.simple import ResNet, train_one_epoch, evaluate
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

    model = ResNet(
        n_classes=train_ds.n_classes,
        coeff=args.coeff,
        n_residuals=args.n_residual,
        spectral_norm=args.coeff != 0
    )
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history_train, history_test = [], []
    for i in range(args.n_epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        history_train.append(train_stats)
        test_stats = evaluate(model, test_loader_id, test_loader_ood, criterion, args.device)
        history_test.append(test_stats)
        print(f"[Ep {i}]", train_stats, test_stats)
        experiment.log_metrics(train_stats, step=i)
        experiment.log_metrics(test_stats, step=i)


def build_dataset(args):
    if args.dataset == 'MNIST':
        transform = Compose([Resize((32, 32)), ToTensor()])
        train_ds = torchvision.datasets.MNIST('data/', train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST('data/', train=False, download=True, transform=transform)

        # Prepare train
        indices_id = (train_ds.targets < 5).nonzero().flatten()
        if args.n_samples:
            indices_id = indices_id[torch.randperm(len(indices_id))[:args.n_samples]]
        train_ds = Subset(train_ds, indices=indices_id)

        indices_id = (test_ds.targets < 5).nonzero().flatten()
        test_ds_id = Subset(test_ds, indices=indices_id)
        indices_ood = (test_ds.targets >= 5).nonzero().flatten()
        test_ds_ood = Subset(test_ds, indices=indices_ood)
        train_ds.n_classes = 5
    elif args.dataset == 'CIFAR':
        # TODO: CIFAR 10 vs CIFAR 100
        raise NotImplementedError
    else:
        raise NotImplementedError

    return train_ds, test_ds_id, test_ds_ood


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--coeff', type=float, default=1)
    parser.add_argument('--n_residual', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=.01)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
