import os
import argparse

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from models import build_model
from utils import plot_grids, write_scalar_dict


def build_model_params(args):
    # TODO
    model_params = {}
    return model_params


def main(args):
    writer = SummaryWriter(log_dir=args.output_dir)

    print(args)
    train_ds, test_ds_id, test_ds_ood, n_classes = build_dataset(args)
    fig = plot_grids(train_ds, test_ds_id, test_ds_ood)
    writer.add_figure('Data Example', fig)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.batch_size*4)
    test_loader_ood = DataLoader(test_ds_ood, batch_size=args.batch_size*4)

    model_params = build_model_params(args)
    model_params.update({'n_samples': len(train_ds), 'n_classes': n_classes})

    model_dict = build_model(args, model_params)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    lr_scheduler = model_dict.get('lr_scheduler')

    history_train, history_test = [], []
    for i_epoch in range(args.n_epochs):
        # Train
        train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
        history_train.append(train_stats)
        if lr_scheduler:
            lr_scheduler.step()
        write_scalar_dict(writer, prefix='train', dict=train_stats, global_step=i_epoch)

        # Eval
        if (i_epoch+1) % args.eval_every == 0:
            test_stats = evaluate(model, test_loader_id, test_loader_ood, **model_dict['eval_kwargs'])
            history_test.append(test_stats)
            write_scalar_dict(writer, prefix='test', dict=test_stats, global_step=i_epoch)
            print(f"Epoch [{i_epoch}]", train_stats, test_stats)

        # Saving checkpoint
        checkpoint = {
            "args": args,
            "model": model.state_dict(),
            "optimizer": model_dict['train_kwargs']['optimizer'].state_dict(),
            "epoch": i_epoch,
            "train_history": history_train,
            "test_history": history_test,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, f"model_{i_epoch}.pth"))
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    print(f"Saving results to {fname}.")
    torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST_vs_MNIST', choices=[
        'MNIST_vs_MNIST',
        'CIFAR10_vs_CIFAR100',
        'CIFAR10_vs_SVHN',
        'CIFAR100_vs_CIFAR10',
        'CIFAR100_vs_SVHN',
    ])
    parser.add_argument('--model', type=str, default='vanilla', choices=['vanilla', 'DDU', 'SNGP', 'sghmc'])
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--coeff', type=float, default=1)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()
    main(args)
