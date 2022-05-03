from comet_ml import Experiment # fmt: off

import argparse

import torch
import torch.nn as nn

import torchvision

from torch.utils.data import DataLoader

from datasets import build_dataset
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
    train_ds, test_ds_id, test_ds_ood, n_classes = build_dataset(args)
    fig = plot_grids(train_ds, test_ds_id, test_ds_ood)
    experiment.log_figure(fig)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.batch_size*4)
    test_loader_ood = DataLoader(test_ds_ood, batch_size=args.batch_size*4)

    model_dict = build_model(args, n_classes)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    lr_scheduler = model_dict.get('lr_scheduler')

    history_train, history_test = [], []
    for i_epoch in range(args.n_epochs):
        train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
        history_train.append(train_stats)
        if lr_scheduler:
            lr_scheduler.step()
        test_stats = evaluate(model, test_loader_id, test_loader_ood, **model_dict['eval_kwargs'])
        history_test.append(test_stats)
        print(f"Epoch [{i_epoch}]", train_stats, test_stats)
        experiment.log_metrics(train_stats, step=i_epoch)
        experiment.log_metrics(test_stats, step=i_epoch)

def build_model(args, n_classes):
    if args.model == 'vanilla':
        model = torchvision.models.resnet18(True)
        model.fc = nn.Linear(512, n_classes)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            momentum=args.momentum
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': vanilla.train_one_epoch,
            'evaluate': vanilla.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': ddu.train_one_epoch,
            'evaluate': ddu.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model == 'SNGP':
        backbone = resnet18(spectral_normalization=(args.coeff != 0), num_classes=n_classes, coeff=args.coeff)
        model = sngp.SNGP(
            backbone,
            in_features=512,
            num_inducing=1024,
            num_classes=n_classes,
            kernel_scale=5,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum, 
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sngp.train_one_epoch,
            'evaluate': sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    else:
        NotImplementedError(f'Model {args.model} not implemented.')

    return model_dict


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

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=.1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
