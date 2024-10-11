import datetime
import json
import logging
import os
import sys
import time

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies.random import RandomSampling
from dal_toolbox.active_learning.strategies import uncertainty, coreset, badge, typiclust, xpal, xpalclust, \
    randomclust, prob_cover, eer, linear_xpal, falcun#, dropquery, alfamix

from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything, _calculate_mean_gamma, kernels


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
    logger.info('Building dataset: %s', args.dataset)
    data = build_dataset(args)

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args.al_strategy.name, args, num_classes=data.num_classes, results=results)

    # Setup Model
    logger.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes)

    # Setup AL Module
    logger.info(f'Creating AL Datamodule with {args.al_cycle.n_init} randomly chosen initial samples.')
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        test_dataset=data.test_dataset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    al_datamodule.random_init(n_samples=args.al_cycle.n_init)

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
        callbacks = [MetricLogger()]
        trainer = L.Trainer(
            max_epochs=args.model.num_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            accelerator='gpu',
            default_root_dir=args.output_dir,
            enable_progress_bar=False,
            check_val_every_n_epoch=args.val_interval,
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logger.info('Evaluation..')
        #TODO: REPLACE AFTER DEBUGGING
        #train_stats, validation_stats, test_stats = evaluate_model(model, trainer, al_datamodule, logger)
        train_stats, validation_stats, test_stats = {}, {}, {}

        cycle_results.update({
            "train_stats": train_stats,
            "validation_stats": validation_stats,
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
        optimizer = torch.optim.SGD(params=model.parameters(), **args.model.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
    else:
        raise NotImplementedError(f"Model {args.model.name} not implemented.")

    model = deterministic.DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return model


def build_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.data_dir)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.data_dir)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.data_dir)
    else:
        raise NotImplementedError(f"Dataset {args.dataset.name} is not implemented!")
    return data


def build_al_strategy(name, args, num_classes=None, train_features=None, results=None):
    subset_size = None if args.al_strategy.subset_size == "None" else args.al_strategy.subset_size

    if name == "random": #tested
        query =  RandomSampling()
    elif name == "entropy": #tested 
        query = uncertainty.EntropySampling(subset_size=subset_size)
    elif name == "coreset": #tested
        query = coreset.CoreSet(subset_size=subset_size)
    elif name == "badge": #tested
        query = badge.Badge(subset_size=subset_size)
    elif name == "typiclust": # tested
        query = typiclust.TypiClust(subset_size=subset_size)
    #elif name == "dropquery":
    #    query = dropquery.DropQuery(subset_size=subset_size)
    #elif name == "alfamix":
    #    query = alfamix.AlfaMix(subset_size=subset_size)
    elif name == "falcun":
        query = falcun.Falcun(subset_size=subset_size, gamma=args.al_strategy.gamma, custom_dist=args.al_strategy.custom_dist, deterministic=args.al_strategy.deterministic)
    elif name == "randomclust": # tested
        query = randomclust.RandomClust(subset_size=subset_size)
    elif name == "xpal" or name == "xpalclust": # TODO: Discuss what to do with these strategies
        if args.al_strategy.kernel.gamma == "calculate":
            gamma = _calculate_mean_gamma(train_features)
        else:
            gamma = args.al_strategy.kernel.gamma
        if args.al_strategy.precomputed:
            S = kernels(X=train_features, Y=train_features, metric=args.al_strategy.kernel.name, gamma=gamma)
        else:
            S = None

        if isinstance(args.al_strategy.alpha, str):
            if not args.al_strategy.precomputed:
                sys.exit("Cannot compute alpha without precomputed S. Set precomputed to True.")
            np.fill_diagonal(S, np.nan)  # Filter out self-similarity
            if args.al_strategy.alpha == "median":
                alpha = np.nanmedian(S)
            elif args.al_strategy.alpha == "mean":
                alpha = np.nanmean(S)
            elif "quantile" in args.al_strategy.alpha:
                q = float(args.al_strategy.alpha.split("_")[1])
                alpha = np.nanquantile(S, q=q)
            else:
                raise NotImplementedError(f"Alpha strategy {args.al_strategy.alpha} is not implemented")
            results["alpha"] = float(alpha)
            np.fill_diagonal(S, 1.0)  # Fill it back in
        else:
            alpha = args.al_strategy.alpha
        print(f"Using alpha = {alpha}")

        if name == "xpal":
            query = xpal.XPAL(num_classes, S, subset_size=subset_size, alpha_c=alpha, alpha_x=alpha,
                              precomputed=args.al_strategy.precomputed, gamma=gamma,
                              kernel=args.al_strategy.kernel.name)
        elif name == "xpalclust":
            query = xpalclust.XPALClust(num_classes, S, subset_size=subset_size, alpha_c=alpha, alpha_x=alpha,
                                        precomputed=args.al_strategy.precomputed, gamma=gamma,
                                        kernel=args.al_strategy.kernel.name)
    elif name == "probcover": #TODO: ERROR
        delta = args.al_strategy.delta
        if delta is None:
            delta = prob_cover.estimate_delta(train_features, num_classes, args.al_strategy.alpha)
            print(f"Using calculated delta={delta:.5f}")
        query = prob_cover.ProbCover(subset_size=subset_size, delta=delta)
    elif name == "eer": #TODO: Requires model that is capeable of mc_forward -> add module
        query = eer.MELL(subset_size=subset_size)
    elif name == "linearxpal": #TODO: Requires Debugging, i think something about datasets in pytorch changed
        query = linear_xpal.LinearXPAL(subset_size=subset_size)
    else:
        raise NotImplementedError(f"Active learning strategy {name} is not implemented!")
    return query


if __name__ == "__main__":
    main()
