import logging
import os
import sys

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dal_toolbox import datasets, metrics
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss
from dal_toolbox.models.utils.base import BaseModule
from dal_toolbox.utils import is_running_on_slurm


def build_ssl(name, args):
    if name == 'simclr':
        return build_simclr(args)
    else:
        raise NotImplementedError(f"{name} is not implemented!")


def build_encoder(name, return_output_dim=False):
    if name == 'resnet18_deterministic':
        encoder = deterministic.resnet.ResNet18(num_classes=1)  # Linear layer will be replaced
        encoder.linear = nn.Identity()  # Replace layer after max pool

        encoder_output_dim = 512  # TODO Load dynamically maybe?
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


# TODO The structure should be solved differently
class LinearEvaluationAccuracy():
    def __init__(self, model: BaseModule, output_dim: int, args):
        data = build_dataset(args)

        # Splitting off projection head
        num_classes = data.num_classes
        encoder = model.encoder
        encoder.num_classes = data.num_classes

        # Freeze encoder
        for name, p in encoder.named_parameters():
            p.requires_grad = False

        head = nn.Linear(output_dim, num_classes)
        model = nn.Sequential(encoder, head)

        optimizer = torch.optim.SGD(head.parameters(), **args.le_model.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.le_model.num_epochs)
        self.model = deterministic.DeterministicModel(
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )
        self.train_dataloader = DataLoader(data.train_dataset,
                                           batch_size=args.le_model.train_batch_size,
                                           num_workers=args.n_cpus,
                                           shuffle=True)

        self.val_dataloader = DataLoader(data.val_dataset,
                                         batch_size=args.le_model.val_batch_size,
                                         num_workers=args.n_cpus,
                                         shuffle=False)
        self.trainer = L.Trainer(
            max_epochs=args.le_model.num_epochs,
            enable_checkpointing=False,
            default_root_dir=args.output_dir,
            enable_progress_bar=True,
            check_val_every_n_epoch=args.le_val_interval,
            enable_model_summary=True,
        )

    def compute(self):
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)


@hydra.main(version_base=None, config_path="../self_supervised_learning/config", config_name="config")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))

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

    logger.info("Starting linear evaluation.")
    # Load best performing SSL model
    encoder, output_dim = build_encoder(args.ssl_model.encoder, True)
    projector = build_projector(args.ssl_model.projector, output_dim, args.ssl_model.projector_dim)  # will be replaced
    model = simclr.SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                               encoder=encoder, projector=projector)
    lr = LinearEvaluationAccuracy(model, output_dim, args)
    lr.compute()


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
