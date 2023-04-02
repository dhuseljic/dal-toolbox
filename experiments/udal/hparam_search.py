import os
import hydra
import logging

import ray
import ray.tune as tune

import torch
import torch.nn as nn

from torch.utils.data import random_split

from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.repeater import Repeater

from dal_toolbox import datasets
from dal_toolbox.models import deterministic, mc_dropout, ensemble
from dal_toolbox.utils import seed_everything


def train(config, args):
    # Overwrite args
    args.random_seed = config['__trial_index__']
    args.model.optimizer.lr = float(config['lr'])
    args.model.optimizer.weight_decay = float(config['weight_decay'])
    if 'mixup_alpha' in config.keys():
        args.model.mixup_alpha = float(config['mixup_alpha'])
    elif 'label_smoothing' in config.keys():
        args.model.label_smoothing = float(config['label_smoothing'])
    elif 'dropout_rate' in config.keys():
        args.model.dropout_rate= float(config['dropout_rate'])

    print("Using model args: {}".format(args.model))

    seed_everything(args.random_seed)

    train_ds, val_ds, ds_info = build_datasets(args)

    trainer = build_model(args, n_classes=ds_info['n_classes'])

    train_indices = torch.randperm(len(train_ds))[:args.budget]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.model.batch_size, sampler=train_indices)
    trainer.train(args.model.n_epochs, train_loader=train_loader)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.model.batch_size)
    test_stats = trainer.evaluate(dataloader=val_loader)

    return test_stats


@hydra.main(version_base=None, config_path="./configs", config_name="hparam_search")
def main(args):
    logger = logging.getLogger()
    logger.info('Using setup: %s', args)
    
    # Setup Search space
    search_space, points_to_evaluate = build_search_space(args)
    search_alg = BayesOptSearch(points_to_evaluate=points_to_evaluate)
    search_alg = Repeater(search_alg, repeat=args.n_reps)
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.n_opt_samples *
                                  args.n_reps, metric="test_nll", mode="min")

    # Setup tuner and objective
    objective = tune.with_resources(train, resources={'cpu': args.cpus_per_trial, 'gpu': args.gpus_per_trial})
    objective = tune.with_parameters(objective, args=args)

    # Init ray, if we are using slurm, set cpu and gpus
    adress = 'auto' if args.distributed else None
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', args.cpus_per_trial))
    num_gpus = torch.cuda.device_count()
    ray.init(address=adress, num_cpus=num_cpus, num_gpus=num_gpus)

    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print('Best NLL Hyperparameter: {}'.format(results.get_best_result()))
    print('Best Acc Hyperparameter: {}'.format(results.get_best_result(metric="test_acc1", mode="max", scope='avg').config))


def build_search_space(args):
    points_to_evaluate = None
    if args.model.name == 'resnet18_deterministic':
        search_space = {
            "lr": tune.uniform(1e-4, .5),
            "weight_decay": tune.uniform(0, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-1, "weight_decay": 5e-4},
            {"lr": 1e-2, "weight_decay": 0.05},
        ]
    elif args.model.name == 'resnet18_labelsmoothing':
        search_space = {
            "lr": tune.uniform(1e-4, .5),
            "weight_decay": tune.uniform(0, .1),
            "label_smoothing": tune.uniform(0, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-1, "weight_decay": 5e-4, 'label_smoothing': 0.05},
            {"lr": 1e-2, "weight_decay": 0.05, 'label_smoothing': 0.05},
        ]
    elif args.model.name == 'resnet18_mixup':
        search_space = {
            "lr": tune.uniform(1e-4, .5),
            "weight_decay": tune.uniform(0, .1),
            "mixup_alpha": tune.uniform(.1, .4),
        }
        points_to_evaluate = [
            {"lr": 1e-1, "weight_decay": 5e-4, 'mixup_alpha': 0.1},
            {"lr": 1e-2, "weight_decay": 0.05, 'mixup_alpha': 0.4},
        ]
    elif args.model.name == 'resnet18_mcdropout':
        search_space = {
            "lr": tune.uniform(1e-4, .5),
            "weight_decay": tune.uniform(0, .1),
            "dropout_rate": tune.uniform(1e-4, .5),
        }
        points_to_evaluate = [
            {"lr": 1e-1, "weight_decay": 5e-4, 'dropout_rate': 0.1},
            {"lr": 1e-2, "weight_decay": 0.05, 'dropout_rate': 0.3},
        ]
    elif args.model.name == 'resnet18_ensemble':
        # We only optimize a single value for all members
        search_space = {
            "lr": tune.uniform(1e-4, .5),
            "weight_decay": tune.uniform(0, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-1, "weight_decay": 5e-4},
            {"lr": 1e-2, "weight_decay": 0.05},
        ]
    else:
        raise NotImplementedError('Model {} not implemented.'.format(args.model.name))
    return search_space, points_to_evaluate


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']

    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        trainer = deterministic.trainer.DeterministicTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir
        )

    elif args.model.name == 'resnet18_labelsmoothing':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        trainer = deterministic.trainer.DeterministicTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir
        )

    elif args.model.name == 'resnet18_mixup':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        trainer = deterministic.trainer.DeterministicMixupTrainer(
            model=model,
            criterion=criterion,
            mixup_alpha=args.model.mixup_alpha,
            n_classes=n_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir
        )
    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(n_classes, args.model.n_passes, args.model.dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        trainer = mc_dropout.trainer.MCDropoutTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir,
        )
    elif args.model.name == 'resnet18_ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = deterministic.resnet.ResNet18(n_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
                nesterov=True
            )
            lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.model.n_epochs)
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = ensemble.voting_ensemble.Ensemble(members)
        criterion = nn.CrossEntropyLoss()
        optimizer = ensemble.voting_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = ensemble.voting_ensemble.EnsembleLRScheduler(lr_schedulers)
        trainer = ensemble.trainer.EnsembleTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir,
        )

    else:
        raise NotImplementedError()

    return trainer


def build_datasets(args):

    if args.dataset.name == 'CIFAR10':
        train_ds, ds_info = datasets.cifar.build_cifar10('train', args.dataset_path, return_info=True)

    elif args.dataset.name == 'CIFAR100':
        train_ds, ds_info = datasets.cifar.build_cifar100('train', args.dataset_path, return_info=True)

    elif args.dataset.name == 'SVHN':
        train_ds, ds_info = datasets.svhn.build_svhn('train', args.dataset_path, return_info=True)

    else:
        raise NotImplementedError('Dataset not available')

    # Random split
    generator = torch.Generator().manual_seed(args.random_seed)
    n_val = int(args.val_split * len(train_ds))
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, lengths=[n_train, n_val], generator=generator)

    return train_ds, val_ds, ds_info


if __name__ == '__main__':
    main()
