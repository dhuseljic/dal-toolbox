import datetime
import json
import logging
import os
import time
import copy

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies.random import RandomSampling
from dal_toolbox.active_learning.strategies import uncertainty, coreset, badge, typiclust, randomclust, falcun, dropquery, alfamix

from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything

from dal_toolbox.datasets.dino import FeatureDataset, DinoTransforms, build_dino_model


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.path.output_dir, exist_ok=True)
    os.makedirs(args.path.storage_dir, exist_ok=True)
    os.makedirs(args.path.cache_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    queried_indices = {}

    # Setup Dataset
    logger.info('Building dataset: %s', args.dataset)
    train_dataset, query_dataset, val_dataset, test_dataset, num_classes = build_dataset(args)

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args.al_strategy.name, args)

    # Setup Model
    logger.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=num_classes)

    # Setup AL Module
    logger.info(f'Creating AL Datamodule with {args.al_cycle.n_init} randomly chosen initial samples.')
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_dataset,
        query_dataset=query_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    
    # Ensure reproducability by loading a predefined al_datamodule
    dm_path = os.path.join(args.path.storage_dir, 'dm_' + args.dataset.name + '_' + str(args.al_cycle.n_init) + '_seed' + str(args.random_seed) + '.json')
    if os.path.exists(dm_path):
        logging.info(f"Loading intial labeled pool from {dm_path}!")
        with open(dm_path, 'r') as f:
            dm_state_dict = json.load(f)
        al_datamodule.load_state_dict(dm_state_dict)
    else:
        al_datamodule.random_init(n_samples=args.al_cycle.n_init)
        logging.info(f"Saving intial labeled pool to {dm_path} for reproducability!")
        dm_state_dict = al_datamodule.state_dict()
        with open(dm_path, 'w') as f:
            json.dump(dm_state_dict, f)

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
        if args.al_cycle.cold_start:
            model.reset_states()

        # Train with updated annotations
        logger.info('Training..')
        callbacks = [MetricLogger()]
        trainer = L.Trainer(
            max_epochs=args.model.num_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            accelerator='gpu',
            default_root_dir=args.path.output_dir,
            enable_progress_bar=False,
            check_val_every_n_epoch=args.val_interval,
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logger.info('Evaluation..')
        train_stats, validation_stats, test_stats = evaluate_model(model, trainer, al_datamodule, logger)

        cycle_results.update({
            "train_stats": train_stats,
            "validation_stats": validation_stats,
            "test_stats": test_stats,
        })
        results[f'cycle{i_acq}'] = cycle_results

    # Saving results
    file_name = os.path.join(args.path.output_dir, 'results.json')
    logger.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)



def evaluate_model(model, trainer, al_datamodule, logger):
    train_predictions = trainer.predict(model, al_datamodule.train_dataloader())
    train_logits = torch.cat([pred[0] for pred in train_predictions])
    train_targets = torch.cat([pred[1] for pred in train_predictions])

    train_stats = {
        'accuracy': metrics.Accuracy()(train_logits, train_targets).item(),
        'nll': torch.nn.CrossEntropyLoss()(train_logits, train_targets).item(),
        'brier': metrics.BrierScore()(train_logits, train_targets).item(),
        'ece': metrics.ExpectedCalibrationError()(train_logits, train_targets).item(),
        'ace': metrics.AdaptiveCalibrationError()(train_logits, train_targets).item(),
    }
    logger.info('Train stats: %s', train_stats)

    validation_predictions = trainer.predict(model, al_datamodule.val_dataloader())
    validation_logits = torch.cat([pred[0] for pred in validation_predictions])
    validation_targets = torch.cat([pred[1] for pred in validation_predictions])

    validation_stats = {
        'accuracy': metrics.Accuracy()(validation_logits, validation_targets).item(),
        'nll': torch.nn.CrossEntropyLoss()(validation_logits, validation_targets).item(),
        'brier': metrics.BrierScore()(validation_logits, validation_targets).item(),
        'ece': metrics.ExpectedCalibrationError()(validation_logits, validation_targets).item(),
        'ace': metrics.AdaptiveCalibrationError()(validation_logits, validation_targets).item(),
    }
    logger.info('Validation stats: %s', validation_stats)

    test_predictions = trainer.predict(model, al_datamodule.test_dataloader())
    test_logits = torch.cat([pred[0] for pred in test_predictions])
    test_targets = torch.cat([pred[1] for pred in test_predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_targets).item(),
        'nll': torch.nn.CrossEntropyLoss()(test_logits, test_targets).item(),
        'brier': metrics.BrierScore()(test_logits, test_targets).item(),
        'ece': metrics.ExpectedCalibrationError()(test_logits, test_targets).item(),
        'ace': metrics.AdaptiveCalibrationError()(test_logits, test_targets).item(),
    }
    logger.info('test stats: %s', test_stats)

    return train_stats, validation_stats, test_stats


def build_model(args, num_classes):
    if args.model.name == 'resnet18':
        model = deterministic.resnet.ResNet18(num_classes=num_classes, imagenethead=("ImageNet" in args.dataset.name))
    elif args.model.name == 'dinov2':
        model = deterministic.linear.LinearModel(in_dimension=args.model.embed_dim, num_classes=num_classes)
    else:
        raise NotImplementedError(f"Model {args.model.name} not implemented.")
    
    # Load predefined weights if possible for ensuring reusability
    initial_state_path = os.path.join(args.path.storage_dir, 'weights_' + args.model.name + '_' + str(num_classes) + '_seed' + str(args.random_seed) + '.pth')
    if os.path.exists(initial_state_path):
        logging.info(f"Loading intial model state from {initial_state_path}!")
        initial_state = torch.load(initial_state_path, weights_only=True)
        model.load_state_dict(initial_state)
    else:
        logging.info(f"Saving intial model state for reproducability to {initial_state_path}!")
        initial_state = copy.deepcopy(model.state_dict())
        torch.save(initial_state, initial_state_path)

    optimizer = torch.optim.SGD(params=model.parameters(), **args.model.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
    model = deterministic.DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return model


def build_dataset(args):
    transforms = DinoTransforms(size=(256, 256)) if args.model.name == 'dinov2' else None

    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.path.data_dir, transforms=transforms)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.path.data_dir, transforms=transforms)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.path.data_dir, transforms=transforms)
    elif args.dataset.name == 'ImageNet':
        data = datasets.ImageNet(args.path.imagenet_dir, transforms=transforms)
    else:
        raise NotImplementedError(f"Dataset {args.dataset.name} is not implemented!")    
        
    
    # Prepare feature extraction beforehand to save time when training linear layer.
    if args.model.name == 'dinov2':
        model = build_dino_model(args)
        full_train_ds, test_ds = data.full_train_dataset, data.test_dataset
        full_train_ds = FeatureDataset(model, full_train_ds, cache=True, cache_dir=args.path.cache_dir)
        test_ds = FeatureDataset(model, test_ds, cache=True, cache_dir=args.path.cache_dir)
        
        # Ensure reproducability by loading a predefined al_datamodule
        ind_path = os.path.join(args.path.storage_dir, 'trainvalsplit_' + args.dataset.name + '_' + str(args.al_cycle.n_init) + '_seed' + str(args.random_seed) + '.json')
        if os.path.exists(ind_path):
            logging.info(f"Loading intial labeled pool from {ind_path}!")
            with open(ind_path, 'r') as f:
                ind = json.load(f)
            train_indices, val_indices = ind['train_indices'], ind['val_indices']
        else:
            train_indices, val_indices = data._get_train_val_indices(len(full_train_ds))
            train_indices, val_indices = train_indices.tolist(), val_indices.tolist()
            logging.info(f"Saving intial labeled pool to {ind_path} for reproducability!")
            ind = {'train_indices' : train_indices, 'val_indices' : val_indices}
            with open(ind_path, 'w') as f:
                json.dump(ind, f)

        train_ds = torch.utils.data.Subset(full_train_ds, indices=train_indices)
        val_ds = torch.utils.data.Subset(full_train_ds, indices=val_indices)
        query_ds = train_ds
    else:
        train_ds, query_ds, val_ds, test_ds = data.train_dataset, data.query_dataset, data.val_dataset, data.test_dataset

    return train_ds, query_ds, val_ds, test_ds, data.num_classes


def build_al_strategy(name, args):
    subset_size = None if args.al_strategy.subset_size == "None" else args.al_strategy.subset_size
    if name == "random":
        query =  RandomSampling()
    elif name == "entropy": 
        query = uncertainty.EntropySampling(subset_size=subset_size)
    elif name == "leastconfidence":
        query = uncertainty.LeastConfidentSampling(subset_size=subset_size)
    elif name == "margin":
        query = uncertainty.MarginSampling(subset_size=subset_size)
    elif name == "coreset":
        query = coreset.CoreSet(subset_size=subset_size)
    elif name == "badge":
        query = badge.Badge(subset_size=subset_size)
    elif name == "typiclust": 
        query = typiclust.TypiClust(subset_size=subset_size)
    elif name == "dropquery":
        query = dropquery.DropQuery(subset_size=subset_size, num_iter=args.al_strategy.n_iter, p_drop=args.al_strategy.p_drop)
    elif name == "alfamix":
        query = alfamix.AlfaMix(subset_size=subset_size, embed_dim=args.model.embed_dim)
    elif name == "falcun":
        query = falcun.Falcun(subset_size=subset_size, gamma=args.al_strategy.gamma, custom_dist=args.al_strategy.custom_dist, deterministic=args.al_strategy.deterministic)
    elif name == "randomclust":
        query = randomclust.RandomClust(subset_size=subset_size)
    else:
        raise NotImplementedError(f"Active learning strategy {name} is not implemented!")
    return query


if __name__ == "__main__":
    main()
