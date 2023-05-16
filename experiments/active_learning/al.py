import os
import time
import json
import logging

import torch
import hydra

import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dal_toolbox import metrics
from dal_toolbox.models import deterministic
from dal_toolbox.datasets import cifar, svhn
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge
from dal_toolbox.utils import seed_everything


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
    train_ds, query_ds, val_ds, ds_info = build_al_datasets(args)
    val_loader = DataLoader(val_ds, batch_size=args.model.predict_batch_size)

    # Setup AL Module
    logging.info('Creating AL Data Module with %s initial samples.', args.al_cycle.n_init)
    al_datamodule = ActiveLearningDataModule(
        train_ds,
        query_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_datamodule.labeled_indices

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=ds_info['n_classes'])

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)
            query_time = time.time() - t1
            logging.info('Querying took %.2f minutes', query_time/60)
            cycle_results['query_indices'] = indices
            cycle_results['query_time'] = query_time
            queried_indices[f'cycle{i_acq}'] = indices

        #  If cold start is set, reset the model parameters
        model.reset_states()

        # Train with updated annotations
        logging.info('Training..')
        trainer = L.Trainer(
            max_epochs=args.model.num_epochs,
            enable_checkpointing=False,
            callbacks=[],
            default_root_dir=args.output_dir,
            fast_dev_run=True
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logging.info('Evaluation..')
        predictions = trainer.predict(model, val_loader)
        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])
        test_stats = {
            'accuracy': metrics.Accuracy()(logits, targets).item(),
            'nll': torch.nn.CrossEntropyLoss()(logits, targets).item(),
            'brier': metrics.BrierScore()(logits, targets).item(),
            'ece': metrics.ExpectedCalibrationError()(logits, targets).item(),
            'ace': metrics.AdaptiveCalibrationError()(logits, targets).item(),
        }
        logging.info('Evaluation stats: %s', test_stats)

        cycle_results.update({
            "test_stats": test_stats,
            "labeled_indices": al_datamodule.labeled_indices,
            "n_labeled_samples": len(al_datamodule.labeled_indices),
            "unlabeled_indices": al_datamodule.unlabeled_indices,
            "n_unlabeled_samples": len(al_datamodule.unlabeled_indices),
        })
        results[f'cycle{i_acq}'] = cycle_results

    # Saving results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Saving indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)


def build_model(args, num_classes):
    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
        model = deterministic.DeterministicModel(
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()}
        )

    return model


def build_al_strategy(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "uncertainty":
        query = uncertainty.EntropySampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "coreset":
        query = coreset.CoreSet(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "badge":
        query = badge.Badge(subset_size=args.al_strategy.subset_size)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


def build_al_datasets(args):
    if args.dataset.name == 'CIFAR10':
        train_ds, ds_info = cifar.build_cifar10('train', args.dataset_path, return_info=True)
        query_ds = cifar.build_cifar10('query', args.dataset_path)
        test_ds = cifar.build_cifar10('test', args.dataset_path)

    elif args.dataset.name == 'CIFAR100':
        train_ds, ds_info = cifar.build_cifar100('train', args.dataset_path, return_info=True)
        query_ds = cifar.build_cifar100('query', args.dataset_path)
        test_ds = cifar.build_cifar100('test', args.dataset_path)

    elif args.dataset.name == 'SVHN':
        train_ds, ds_info = svhn.build_svhn('train', args.dataset_path, return_info=True)
        query_ds = svhn.build_svhn('query', args.dataset_path)
        test_ds = svhn.build_svhn('test', args.dataset_path)
    return train_ds, query_ds, test_ds, ds_info


if __name__ == "__main__":
    main()
