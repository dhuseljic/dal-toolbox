import os
import time
import json
import logging

import torch
import torch.nn as nn
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.utils import seed_everything
from dal_toolbox import datasets
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    queried_indices = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets.')
    train_ds, query_ds, val_ds, ds_info = build_datasets(args)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    al_dataset = ALDataset(train_ds, query_ds, random_state=args.random_seed)
    if args.al_cycle.init_pool_file is not None:
        logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
        with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
            initial_indices = json.load(f)
        assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
        al_dataset.update_annotations(initial_indices)
    else:
        logging.info('Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
        al_dataset.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_dataset.labeled_indices

    if args.ood_datasets:
        logging.info('Building ood datasets.')
        ood_datasets = build_ood_datasets(args)
        ood_loaders = {name: DataLoader(ds, batch_size=args.val_batch_size) for name, ds in ood_datasets.items()}
    else:
        ood_loaders = None

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_model(args, n_classes=ds_info['n_classes'])

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args, device=args.device)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=trainer.model,
                dataset=al_dataset.query_dataset,
                unlabeled_indices=al_dataset.unlabeled_indices,
                labeled_indices=al_dataset.labeled_indices,
                acq_size=args.al_cycle.acq_size
            )
            al_dataset.update_annotations(indices)
            query_time = time.time() - t1
            logging.info('Querying took %.2f minutes', query_time/60)
            cycle_results['query_indices'] = indices
            cycle_results['query_time'] = query_time
            queried_indices[f'cycle{i_acq}'] = indices

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        iter_per_epoch = len(al_dataset.labeled_dataset) // args.model.batch_size + 1
        train_sampler = RandomSampler(al_dataset.labeled_dataset, num_samples=args.model.batch_size*iter_per_epoch)
        train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, sampler=train_sampler)

        trainer.reset_states(reset_model=args.al_cycle.cold_start)
        history = trainer.train(args.model.n_epochs, train_loader=train_loader)
        cycle_results['train_history'] = history['train_history']

        # Evaluate resulting model
        test_stats = trainer.evaluate(val_loader, dataloaders_ood=ood_loaders)
        cycle_results['test_stats'] = test_stats

        # Log
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test_stats/{key}", scalar_value=value, global_step=i_acq)

        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })
        cycle_results.keys()
        results[f'cycle{i_acq}'] = cycle_results

    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Save indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)


def build_query(args, **kwargs):
    device = kwargs.get('device', 'cuda')
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    # Aleatoric Strategies
    elif args.al_strategy.name == "least_confident":
        query = uncertainty.LeastConfidentSampling(
            batch_size=args.model.batch_size,
            subset_size=args.al_strategy.subset_size,
            random_seed=args.random_seed,
            device=device,
        )
    elif args.al_strategy.name == "margin":
        query = uncertainty.MarginSampling(
            batch_size=args.model.batch_size,
            subset_size=args.al_strategy.subset_size,
            random_seed=args.random_seed,
            device=device,
        )
    elif args.al_strategy.name == "entropy":
        query = uncertainty.EntropySampling(
            batch_size=args.model.batch_size,
            subset_size=args.al_strategy.subset_size,
            random_seed=args.random_seed,
            device=device,
        )
    # Epistemic Strategies
    elif args.al_strategy.name == "bayesian_entropy":
        query = uncertainty.BayesianEntropySampling(
            batch_size=args.model.batch_size,
            subset_size=args.al_strategy.subset_size,
            random_seed=args.random_seed,
            device=device,
        )
    elif args.al_strategy.name == 'variation_ratio':
        query = uncertainty.VariationRatioSampling(
            batch_size=args.model.batch_size,
            subset_size=args.al_strategy.subset_size,
            random_seed=args.random_seed,
            device=device,
        )
    elif args.al_strategy.name == 'bald':
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    elif args.al_strategy.name == "coreset":
        device = kwargs['device']
        query = coreset.CoreSet(subset_size=args.al_strategy.subset_size, device=device)
    elif args.al_strategy.name == "badge":
        device = kwargs['device']
        query = badge.Badge(subset_size=args.al_strategy.subset_size, device=device)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


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
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
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
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
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
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
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
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
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
            lrs = CosineAnnealingLRLinearWarmup(opt, num_epochs=args.model.n_epochs, warmup_epochs=10)
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

    elif args.model.name == 'resnet18_sngp':
        model = sngp.resnet.resnet18_sngp(
            num_classes=n_classes,
            input_shape=(3, 32, 32),
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.norm_bound,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=False,
            random_feature_type=args.model.gp.random_feature_type,
            scale_random_features=args.model.gp.scale_random_features,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
        trainer = sngp.trainer.SNGPTrainer(
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
        query_ds = datasets.cifar.build_cifar10('query', args.dataset_path)
        test_ds_id = datasets.cifar.build_cifar10('test', args.dataset_path)

    elif args.dataset.name == 'CIFAR100':
        train_ds, ds_info = datasets.cifar.build_cifar100('train', args.dataset_path, return_info=True)
        query_ds = datasets.cifar.build_cifar100('query', args.dataset_path)
        test_ds_id = datasets.cifar.build_cifar100('test', args.dataset_path)

    elif args.dataset.name == 'SVHN':
        train_ds, ds_info = datasets.svhn.build_svhn('train', args.dataset_path, return_info=True)
        query_ds = datasets.svhn.build_svhn('query', args.dataset_path)
        test_ds_id = datasets.svhn.build_svhn('test', args.dataset_path)

    else:
        raise NotImplementedError('Dataset not available')

    return train_ds, query_ds, test_ds_id, ds_info


def build_ood_datasets(args):

    ood_datasets = {}
    for ds_name in args.ood_datasets:
        if ds_name == 'CIFAR10':
            ood_ds = datasets.cifar.build_cifar10('test', args.dataset_path)
        elif ds_name == 'CIFAR100':
            ood_ds = datasets.cifar.build_cifar100('test', args.dataset_path)
        elif ds_name == 'SVHN':
            ood_ds = datasets.svhn.build_svhn('test', args.dataset_path)
        else:
            raise NotImplementedError(f'Dataset {ds_name} not implemented.')
        ood_datasets[ds_name] = ood_ds

    return ood_datasets


if __name__ == "__main__":
    main()
