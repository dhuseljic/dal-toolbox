import os
import json
import logging

import torch
import torch.nn as nn
import lightning as L
import hydra

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox import metrics
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

    # Setup Dataset
    logging.info('Building datasets.')
    data = build_datasets(args)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        val_dataset=data.val_dataset,
        query_dataset=data.query_dataset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    test_loader = DataLoader(data.test_dataset, batch_size=args.model.predict_batch_size)
    if args.ood_datasets:
        logging.info('Building ood datasets.')
        ood_datasets = build_ood_datasets(args, id_mean=data.mean, id_std=data.std)
        ood_loaders = {name: DataLoader(ds, batch_size=args.model.predict_batch_size)
                       for name, ds in ood_datasets.items()}
    else:
        ood_loaders = None

    if args.al_cycle.init_pool_file is not None:
        logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
        with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
            initial_indices = json.load(f)
        assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
        al_datamodule.update_annotations(initial_indices)
    else:
        logging.info('Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
        al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_datamodule.labeled_indices

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes)

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)
            queried_indices[f'cycle{i_acq}'] = indices


        # Train
        model.reset_states(reset_model_parameters=args.al_cycle.cold_start)
        trainer = L.Trainer(
            max_epochs=args.model.n_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            logger=False,
            check_val_every_n_epoch=args.val_every,
            enable_progress_bar=(not is_running_on_slurm()),
            callbacks=[MetricLogger()] if is_running_on_slurm() else [],
            fast_dev_run=args.fast_dev_run
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        predictions = trainer.predict(model, test_loader)
        logits = torch.cat([preds[0] for preds in predictions])
        targets = torch.cat([preds[1] for preds in predictions])
        test_stats = evaluate(logits, targets)
        for name, loader in ood_loaders.items():
            predictions_ood = trainer.predict(model, loader)
            logits_ood = torch.cat([preds[0] for preds in predictions_ood])
            ood_stats = evaluate_ood(logits, logits_ood)
            ood_stats = {f'{key}_{name}': val for key, val in ood_stats.items()}
            test_stats.update(ood_stats)
        logging.info("[Acq %s] Test statistics: %s", i_acq, test_stats)

        cycle_results['test_stats'] = test_stats
        cycle_results.update({
            "labeled_indices": al_datamodule.labeled_indices,
            "n_labeled_samples": len(al_datamodule.labeled_indices),
            "unlabeled_indices": al_datamodule.unlabeled_indices,
            "n_unlabeled_samples": len(al_datamodule.unlabeled_indices),
        })
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


def evaluate(logits, targets):
    test_stats = {}
    if logits.ndim == 3:
        logits = metrics.ensemble_log_softmax(logits)
    test_stats["accuracy"] = metrics.Accuracy()(logits, targets).item()
    test_stats["nll"] = torch.nn.CrossEntropyLoss()(logits, targets).item()
    test_stats["brier"] = metrics.BrierScore()(logits, targets).item()
    test_stats["tce"] = metrics.ExpectedCalibrationError()(logits, targets).item()
    test_stats["ace"] = metrics.AdaptiveCalibrationError()(logits, targets).item()
    return test_stats


def evaluate_ood(logits_id, logits_ood):
    test_stats = {}

    if logits_id.ndim == 2:
        entropy_id = metrics.entropy_from_logits(logits_id)
        entropy_ood = metrics.entropy_from_logits(logits_ood)
    else:
        entropy_id = metrics.ensemble_entropy_from_logits(logits_id)
        entropy_ood = metrics.ensemble_entropy_from_logits(logits_ood)

    test_stats["aupr"] = metrics.OODAUPR()(entropy_id, entropy_ood).item()
    test_stats["auroc"] = metrics.OODAUROC()(entropy_id, entropy_ood).item()
    return test_stats


def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "least_confident":
        query = uncertainty.LeastConfidentSampling(subset_size=args.al_strategy.subset_size,)
    elif args.al_strategy.name == "margin":
        query = uncertainty.MarginSampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "entropy":
        query = uncertainty.EntropySampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "bayesian_entropy":
        query = uncertainty.BayesianEntropySampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == 'variation_ratio':
        query = uncertainty.VariationRatioSampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == 'bald':
        query = uncertainty.BALDSampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "coreset":
        query = coreset.CoreSet(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "badge":
        query = badge.Badge(subset_size=args.al_strategy.subset_size)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


def build_model(args, **kwargs):
    num_classes = kwargs['num_classes']

    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
        model = deterministic.DeterministicModel(
            model, criterion, optimizer, lr_scheduler,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()}
        )
        return model
    elif args.model.name == 'resnet18_labelsmoothing':
        model = deterministic.resnet.ResNet18(num_classes)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
        model = deterministic.DeterministicModel(
            model, criterion, optimizer, lr_scheduler,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()}
        )
    elif args.model.name == 'resnet18_mixup':
        model = deterministic.resnet.ResNet18(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
        model = deterministic.DeterministicMixupModel(
            model, num_classes, args.model.mixup_alpha, criterion, optimizer, lr_scheduler,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()}
        )
    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(num_classes, args.model.n_passes, args.model.dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.n_epochs, warmup_epochs=10)
        model = mc_dropout.MCDropoutModel(
            model, criterion, optimizer, lr_scheduler,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()}
        )
    elif args.model.name == 'resnet18_ensemble':
        members, lr_scheduler_list, optimizer_list = [], [], []
        for _ in range(args.model.n_member):
            mem = deterministic.resnet.ResNet18(num_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
                nesterov=True
            )
            lrs = CosineAnnealingLRLinearWarmup(opt, num_epochs=args.model.n_epochs, warmup_epochs=10)
            members.append(mem)
            optimizer_list.append(opt)
            lr_scheduler_list.append(lrs)
        criterion = nn.CrossEntropyLoss()
        model = ensemble.EnsembleModel(
            members, criterion, optimizer_list, lr_scheduler_list,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()},
        )
    elif args.model.name == 'resnet18_sngp':
        model = sngp.resnet.resnet18_sngp(
            num_classes=num_classes,
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
        model = sngp.SNGPModel(
            model, criterion, optimizer, lr_scheduler,
            {'train_acc': metrics.Accuracy()}, {'val_acc': metrics.Accuracy()},
        )
    else:
        raise NotImplementedError()

    return model


def build_datasets(args):
    if args.dataset == 'CIFAR10':
        data = datasets.cifar.CIFAR10(args.dataset_path)
    elif args.dataset == 'CIFAR100':
        data = datasets.cifar.CIFAR100(args.dataset_path)
    elif args.dataset == 'SVHN':
        data = datasets.svhn.SVHN(args.dataset_path)
    else:
        raise NotImplementedError('Dataset not available')

    return data


def build_ood_datasets(args, id_mean, id_std):
    ood_datasets = {}
    for ds_name in args.ood_datasets:
        if ds_name == 'CIFAR10':
            data = datasets.cifar.CIFAR10(args.dataset_path, mean=id_mean, std=id_std)
        elif ds_name == 'CIFAR100':
            data = datasets.cifar.CIFAR100(args.dataset_path, mean=id_mean, std=id_std)
        elif ds_name == 'SVHN':
            data = datasets.svhn.SVHN(args.dataset_path, mean=id_mean, std=id_std)
        else:
            raise NotImplementedError(f'Dataset {ds_name} not implemented.')
        ood_datasets[ds_name] = data.test_dataset

    return ood_datasets


if __name__ == "__main__":
    main()
