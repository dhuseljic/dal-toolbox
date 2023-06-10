import datetime
import json
import logging
import os
import sys
import time

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dal_toolbox import datasets, metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import is_running_on_slurm
from experiments.active_learning.active_learning import build_al_strategy


def build_ssl(name, args):
    if name == 'simclr':
        return build_simclr(args)
    else:
        raise NotImplementedError(f"{name} is not implemented!")


def build_encoder(name, return_output_dim=False):
    if name == 'resnet18_deterministic':
        encoder = deterministic.resnet.ResNet18(num_classes=1)  # Linear layer will be replaced
        encoder.linear = nn.Identity()  # Replace layer after max pool

        encoder_output_dim = 512
    else:
        raise NotImplementedError(f"{name} is not implemented!")

    if return_output_dim:
        return encoder, encoder_output_dim
    return encoder


def build_projector(name, input_dim, output_dim):
    if name == 'mlp':
        projector = nn.Sequential(nn.Linear(input_dim, output_dim),
                                  nn.ReLU(),
                                  nn.Linear(output_dim, output_dim))
    else:
        raise NotImplementedError(f"{name} is not implemented!")
    return projector


def build_simclr(args) -> (nn.Module, nn.Module):
    encoder, encoder_output_dim = build_encoder(args.ssl_model.encoder, True)
    projector = build_projector(args.ssl_model.projector, encoder_output_dim, args.ssl_model.projector_dim)

    optimizer = torch.optim.AdamW(
        params=list(encoder.parameters()) + list(projector.parameters()),
        lr=args.ssl_model.optimizer.lr,
        weight_decay=args.ssl_model.optimizer.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.ssl_model.n_epochs,
        eta_min=args.ssl_model.optimizer.lr / 50
    )

    model = simclr.SimCLR(
        encoder=encoder,
        projector=projector,
        loss_fn=InfoNCELoss(args.ssl_model.temperature),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc_top1': metrics.ContrastiveAccuracy(),
                       'train_acc_top5': metrics.ContrastiveAccuracy(topk=5)},
        val_metrics={'val_acc_top1': metrics.ContrastiveAccuracy(),
                     'val_acc_top5': metrics.ContrastiveAccuracy(topk=5)}
    )
    model.encoder_output_dim = encoder_output_dim

    return model


@hydra.main(version_base=None, config_path="../ssl_al/config", config_name="config")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))

    # Necessary for logging
    results = {}
    queried_indices = {}

    # To be reproducable
    L.seed_everything(args.random_seed)

    # Create a Data Module
    data = build_contrastive_dataset(args)
    train_dataloader = DataLoader(data.train_dataset,
                                  batch_size=args.ssl_model.train_batch_size,
                                  num_workers=args.n_cpus,
                                  shuffle=True)

    val_dataloader = DataLoader(data.val_dataset,
                                batch_size=args.ssl_model.val_batch_size,
                                num_workers=args.n_cpus,
                                shuffle=False)

    model = build_ssl(args.ssl_model.name, args)

    # Create a Trainer Module
    trainer = L.Trainer(
        default_root_dir=os.path.join(args.cp_path, "SimCLR"),
        accelerator="auto",
        max_epochs=args.ssl_model.n_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc_top5", every_n_epochs=1),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=args.ssl_val_interval,
        enable_progress_bar=is_running_on_slurm() is False,
    )

    logger.info("Training SSL")
    # Train and automatically save top 5 models based on validation accuracy
    trainer.fit(model, train_dataloader, val_dataloader)

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args)

    # Setup Dataset (Now without contrastive transforms)
    logger.info('Building datasets.')
    data = build_dataset(args)
    test_loader = DataLoader(data.test_dataset, batch_size=args.al_model.predict_batch_size)

    # Setup AL Module
    logger.info('Creating AL Datamodule with %s initial samples.', args.al_cycle.n_init)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        train_batch_size=args.al_model.train_batch_size,
        predict_batch_size=args.al_model.predict_batch_size,
    )
    al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_datamodule.labeled_indices

    logger.info("Preparing AL model.")
    # Load best performing SSL model
    encoder, output_dim = build_encoder(args.ssl_model.encoder, True)
    projector = build_projector(args.ssl_model.projector, output_dim, args.ssl_model.projector_dim)
    model = simclr.SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                               encoder=encoder, projector=projector)

    # Splitting off projection head
    num_classes = data.num_classes
    encoder = model.encoder
    encoder.num_classes = data.num_classes

    # Freeze encoder
    for name, p in encoder.named_parameters():
        p.requires_grad = False

    head = nn.Linear(output_dim, num_classes)
    model = nn.Sequential(encoder, head)

    optimizer = torch.optim.SGD(model.parameters(), **args.al_model.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.al_model.num_epochs)
    model = deterministic.DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy()},
    )

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
            max_epochs=args.al_model.num_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=args.output_dir,
            enable_progress_bar=is_running_on_slurm() is False,
            check_val_every_n_epoch=args.al_val_interval,
            enable_model_summary=i_acq == 0,
            num_sanity_val_steps=0 if i_acq > 0 else 2,
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        logger.info('Evaluation..')
        predictions = trainer.predict(model, test_loader)
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


def build_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.dataset_path)
    else:
        sys.exit(f"Dataset {args.dataset.name} not implemented.")

    return data


def build_contrastive_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10Contrastive(args.dataset_path)
    else:
        sys.exit(f"Dataset {args.dataset.name} not implemented.")

    return data


if __name__ == '__main__':
    main()
