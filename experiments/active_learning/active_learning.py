import datetime
import json
import logging
import os
import sys
import time

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge, typiclust
from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything, is_running_on_slurm


class FeatureDataset(Dataset):
    def __init__(self, model, dataset, device):
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)
        features, labels = model.get_representations(dataloader, device, return_labels=True)
        self.features = features.detach()
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    queried_indices = {}

    # Setup Dataset
    logger.info('Building datasets.')
    if args.precomputed_features:
        features = torch.load(args.precomputed_features_dir)

        trainset = features['trainset']
        queryset = trainset
        valset = features['valset']
        testset = features['testset']

        num_classes = len(torch.unique(trainset.labels))
        feature_size = trainset.features.shape[1]
    else:
        data = build_datasets(args)
        trainset = data.train_dataset
        queryset = data.query_dataset
        valset = data.val_dataset
        testset = data.test_dataset

        num_classes = data.num_classes
        feature_size = None

    # TODO: For some reason the test dataloader of al_datamodule does not work
    test_dataloader = DataLoader(testset, batch_size=args.model.predict_batch_size, shuffle=False)

    # Setup AL Module
    logger.info('Creating AL Datamodule with %s initial samples.', args.al_cycle.n_init)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=trainset,
        query_dataset=queryset,
        val_dataset=valset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_datamodule.labeled_indices

    # Setup Model
    logger.info('Building model: %s', args.model.name)

    model = build_model(args, num_classes=num_classes, feature_size=feature_size)

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logger.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        if i_acq != 0:
            t1 = time.time()
            logger.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)
            query_eta = datetime.timedelta(seconds=int(time.time() - t1))
            logger.info('Querying took %s', query_eta)
            queried_indices[f'cycle{i_acq}'] = indices

        #  model cold start
        model.reset_states()

        # Train with updated annotations
        logger.info('Training..')
        callbacks = []
        if is_running_on_slurm():
            callbacks.append(MetricLogger())
        trainer = L.Trainer(
            max_epochs=args.model.num_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=args.output_dir,
            enable_progress_bar=is_running_on_slurm() is False,
            check_val_every_n_epoch=args.val_interval,
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logger.info('Evaluation..')
        predictions = trainer.predict(model, test_dataloader)
        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])
        test_stats = {
            'accuracy': metrics.Accuracy()(logits, targets).item(),
            'nll': torch.nn.CrossEntropyLoss()(logits, targets).item(),
            'brier': metrics.BrierScore()(logits, targets).item(),
            'ece': metrics.ExpectedCalibrationError()(logits, targets).item(),
            'ace': metrics.AdaptiveCalibrationError()(logits, targets).item(),
        }
        logger.info('Evaluation stats: %s', test_stats)

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
    logger.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Saving indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logger.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)


def build_model(args, num_classes, feature_size=None):
    model = None
    if args.precomputed_features:
        if args.model.name == "linear":
            model = nn.Linear(feature_size, num_classes)
            optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
    else:
        if args.model.name == 'resnet18_deterministic':
            model = deterministic.resnet.ResNet18(num_classes=num_classes)
            optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)

    if model is None:
        sys.exit(f"Model {args.model.name} is not implemented, "
                 f"or not compatible with precomputed features == {args.precomputed_features}")

    model = deterministic.DeterministicModel(
        model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy()},
    )

    return model


def build_al_strategy(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "entropy":
        query = uncertainty.EntropySampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "coreset":
        query = coreset.CoreSet(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "badge":
        query = badge.Badge(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "typiclust":
        query = typiclust.TypiClust(subset_size=args.al_strategy.subset_size)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


def build_datasets(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)

    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)

    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.dataset_path)

    return data


if __name__ == "__main__":
    main()
