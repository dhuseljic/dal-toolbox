import datetime
import gc
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
from sklearn.preprocessing import StandardScaler
from torch import nn

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge, typiclust, xpal, xpalclust, \
    randomclust, prob_cover, eer, linear_xpal
# noinspection PyUnresolvedReferences
from dal_toolbox.datasets.utils import FeatureDataset, FeatureDatasetWrapper
from dal_toolbox.metrics import ensemble_log_softmax
from dal_toolbox.models import deterministic, mc_dropout
from dal_toolbox.models.deterministic.linear import LinearModel
from dal_toolbox.models.parzen_window_classifier import PWCLightning
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything, is_running_on_slurm, kernels, _calculate_mean_gamma


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
    if args.precomputed_features and args.fintuning:
        raise RuntimeError("Cannot use precomputed features and finetuning at the same times")
    if args.precomputed_features:
        data = FeatureDatasetWrapper(args.precomputed_features_dir)
        feature_size = data.num_features

        if args.standardize_precomputed_features:
            train_features = data.train_dataset.dataset.features
            scaler = StandardScaler().fit(train_features)

            data.train_dataset.dataset.features = torch.from_numpy(scaler.transform(train_features)).to(
                dtype=torch.float32)
            data.val_dataset.dataset.features = torch.from_numpy(
                scaler.transform(data.val_dataset.dataset.features)).to(dtype=torch.float32)
            data.test_dataset.features = torch.from_numpy(scaler.transform(data.test_dataset.features)).to(
                dtype=torch.float32)

        features = torch.stack([batch[0] for batch in data.train_dataset])
    else:
        data = build_dataset(args)
        feature_size = None
        features = None  # This takes up too much memory otherwise

    trainset = data.train_dataset
    queryset = data.query_dataset
    valset = data.val_dataset
    testset = data.test_dataset

    num_classes = data.num_classes

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args.al_strategy.name, args, num_classes=num_classes, train_features=features,
                                    results=results)

    # Setup Model
    logger.info('Building model: %s', args.model.name)

    if args.model.name == "parzen_window":
        accelerator = "cpu"
        if args.model.kernel.gamma == "calculate":
            args.model.kernel.gamma = _calculate_mean_gamma(features)
            results["gamma"] = args.model.kernel.gamma
            logger.info(f"Calculated gamma as {args.model.kernel.gamma}.")
    else:
        # accelerator = "cpu"
        accelerator = "auto"

    if args.finetuning:
        model = build_finetunig_model(args, num_classes=num_classes)
    else:
        model = build_model(args, num_classes=num_classes, feature_size=feature_size)

    # Setup AL Module
    logger.info(f'Creating AL Datamodule with {args.al_cycle.n_init} initial samples, '
                f'chosen with strategy {args.al_cycle.init_strategy}.')
    al_datamodule = ActiveLearningDataModule(
        train_dataset=trainset,
        query_dataset=queryset,
        val_dataset=valset,
        test_dataset=testset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    if args.al_cycle.init_strategy == "random":
        al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    else:
        init_al_strategy = build_al_strategy(args.al_cycle.init_strategy, args, num_classes, train_features=features,
                                             results=results)
        indices = init_al_strategy.query(
            model=model,
            al_datamodule=al_datamodule,
            acq_size=args.al_cycle.n_init
        )
        al_datamodule.update_annotations(indices)
    queried_indices['cycle0'] = al_datamodule.labeled_indices
    gc.collect()  # init_al_strategy sometimes takes too long to be automatically collected

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
            accelerator=accelerator,
            default_root_dir=args.output_dir,
            enable_progress_bar=is_running_on_slurm() is False,
            check_val_every_n_epoch=args.val_interval,
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logger.info('Evaluation..')
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

    # Saving indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logger.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)


def build_finetunig_model(args, num_classes):
    model_params = torch.load(args.finetuning_dir)['model']

    if args.model.name == 'resnet18_deterministic':
        encoder = deterministic.resnet.ResNet18(num_classes=num_classes, imagenethead=("ImageNet" in args.dataset.name))
        encoder_output_dim = 512
    elif args.model.name == "resnet50_deterministic":
        encoder = mc_dropout.resnet.DropoutResNet18(num_classes=num_classes, n_passes=args.model.n_passes,
                                                    dropout_rate=args.model.dropout_rate)
        encoder_output_dim = 2048
    elif args.model.name == "wideresnet2810_deterministic":
        encoder = deterministic.wide_resnet.wide_resnet_28_10(num_classes=num_classes,
                                                              dropout_rate=args.model.dropout_rate,
                                                              imagenethead=("ImageNet" in args.dataset.name))
        encoder_output_dim = 640
    else:
        raise NotImplementedError(f"Model {args.model.name} not implemented.")

    encoder.linear = nn.Identity()
    encoder.load_state_dict(model_params)
    encoder.linear = nn.Linear(encoder_output_dim, num_classes)

    linear_params = [p for p in encoder.linear.parameters()]
    base_params = [p[1] for p in encoder.named_parameters() if p[0] not in ["linear.weight", "linear.bias"]]

    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': args.finetuning_lr},
        {'params': linear_params}
    ], **args.model.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)

    model = deterministic.DeterministicModel(
        encoder, optimizer=optimizer, lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy()},
    )

    return model


def build_model(args, num_classes, feature_size=None):
    model = None
    optimizer = None
    lr_scheduler = None

    if args.precomputed_features:
        if args.model.name == "linear":
            model = deterministic.linear.LinearModel(feature_size, num_classes)
            optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
        elif args.model.name == "parzen_window":
            model = PWCLightning(n_classes=num_classes,
                                 random_state=args.random_seed,
                                 kernel_params=args.model,
                                 train_metrics={'train_acc': metrics.Accuracy()},
                                 val_metrics={'val_acc': metrics.Accuracy()})
            return model
    else:
        if args.model.name == 'resnet18_deterministic':
            model = deterministic.resnet.ResNet18(num_classes=num_classes,
                                                  imagenethead=("ImageNet" in args.dataset.name))
            optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
        elif args.model.name == "resnet18_mcdropout":
            model = mc_dropout.resnet.DropoutResNet18(num_classes=num_classes, n_passes=args.model.n_passes,
                                                      dropout_rate=args.model.dropout_rate)
            optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)

            model = mc_dropout.MCDropoutModel(
                model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                train_metrics={'train_acc': metrics.Accuracy()},
                val_metrics={'val_acc': metrics.Accuracy()}
            )
            return model
        elif args.model.name == "wideresnet2810_deterministic":
            model = deterministic.wide_resnet.wide_resnet_28_10(num_classes=num_classes,
                                                                dropout_rate=args.model.dropout_rate,
                                                                imagenethead=("ImageNet" in args.dataset.name))
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


def build_al_strategy(name, args, num_classes=None, train_features=None, results=None):
    subset_size = None if args.al_strategy.subset_size == "None" else args.al_strategy.subset_size

    if name == "random":
        query = random.RandomSampling()
    elif name == "entropy":
        query = uncertainty.EntropySampling(subset_size=subset_size)
    elif name == "coreset":
        query = coreset.CoreSet(subset_size=subset_size)
    elif name == "badge":
        query = badge.Badge(subset_size=subset_size)
    elif name == "typiclust":
        query = typiclust.TypiClust(subset_size=subset_size)
    elif name == "randomclust":
        query = randomclust.RandomClust(subset_size=subset_size)
    elif name == "xpal" or name == "xpalclust":
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
    elif name == "probcover":
        delta = args.al_strategy.delta
        if delta is None:
            delta = prob_cover.estimate_delta(train_features, num_classes, args.al_strategy.alpha)
            print(f"Using calculated delta={delta:.5f}")
        query = prob_cover.ProbCover(subset_size=subset_size, delta=delta)
    elif name == "eer":
        query = eer.MELL(subset_size=subset_size)
    elif name == "linearxpal":
        query = linear_xpal.LinearXPAL(subset_size=subset_size)
    else:
        raise NotImplementedError(f"Active learning strategy {name} is not implemented!")
    return query


def build_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.dataset_path)
    elif args.dataset.name == 'ImageNet':
        data = datasets.ImageNet(args.dataset_path)
    elif args.dataset.name == 'ImageNet50':
        data = datasets.ImageNet50(args.dataset_path)
    elif args.dataset.name == 'ImageNet100':
        data = datasets.ImageNet100(args.dataset_path)
    elif args.dataset.name == 'ImageNet200':
        data = datasets.ImageNet200(args.dataset_path)
    else:
        raise NotImplementedError(f"Dataset {args.dataset.name} is not implemented!")

    return data


if __name__ == "__main__":
    main()
