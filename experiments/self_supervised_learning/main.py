import logging
import math
import os
import sys

import hydra
import lightning as L
import torch.cuda
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from pytorch_optimizer import LARS
from torch import nn
from torch.utils.data import DataLoader

from dal_toolbox import datasets, metrics
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss, LinearEvaluationAccuracy
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
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

        encoder_output_dim = 512  # TODO (ynagel) Load dynamically maybe?
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

    optimizer = LARS(
        params=list(encoder.parameters()) + list(projector.parameters()),
        lr=args.ssl_model.optimizer.base_lr * math.sqrt(args.ssl_model.train_batch_size),
        weight_decay=args.ssl_model.optimizer.weight_decay,
    )

    lr_scheduler = CosineAnnealingLRLinearWarmup(
        optimizer=optimizer,
        num_epochs=args.ssl_model.n_epochs,
        warmup_epochs=10
    )

    model = simclr.SimCLR(
        encoder=encoder,
        projector=projector,
        log_on_epoch_end=is_running_on_slurm(),
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


class LinearEvaluationCallback(Callback):
    def __init__(self, device, args, logger=None):
        self.args = args
        self.device = device
        self.interval = args.le_model.callback.interval
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.interval == 0:
            if self.logger is not None:
                self.logger.info("Starting linear evaluation callback.")
            lr = LinearEvaluationAccuracy(model=pl_module,
                                          data=build_plain_dataset(self.args),
                                          device=self.device,
                                          train_batch_size=self.args.le_model.train_batch_size,
                                          val_batch_size=self.args.le_model.val_batch_size,
                                          n_cpus=self.args.n_cpus,
                                          base_lr=self.args.le_model.optimizer.base_lr,
                                          weight_decay=self.args.le_model.optimizer.weight_decay,
                                          momentum=self.args.le_model.optimizer.momentum,
                                          epochs=self.args.le_model.num_epochs,
                                          val_interval=self.args.le_val_interval,
                                          output_dir=os.path.join(
                                              self.args.output_dir, "SimCLR_Linear_Evaluation_Callback"),
                                          progressbar=is_running_on_slurm() is False,
                                          )
            if self.logger is not None:
                self.logger.info(f"Epoch {trainer.current_epoch} - "
                                 f"Linear evaluation accuracy: "
                                 f"Train : {lr.compute_accuracy():4f} , Validation: {lr.compute_accuracy('val'):4f}")


@hydra.main(version_base=None, config_path="../self_supervised_learning/config", config_name="config")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # To be reproducible
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

    callbacks = [
        ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc_top5", every_n_epochs=1),
        LearningRateMonitor("epoch"),
    ]

    if args.le_model.callback.enabled:
        callbacks.append(LinearEvaluationCallback(model.encoder_output_dim, args, logger))

    # Create a Trainer Module
    trainer = L.Trainer(
        default_root_dir=os.path.join(args.output_dir, "SimCLR"),
        accelerator="auto",
        max_epochs=args.ssl_model.n_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=args.ssl_val_interval,
        enable_progress_bar=is_running_on_slurm() is False,
    )

    logger.info("Training SSL")
    trainer.fit(model, train_dataloader, val_dataloader)

    logger.info("Starting linear evaluation.")
    # Load best performing SSL model
    encoder, output_dim = build_encoder(args.ssl_model.encoder, True)
    projector = build_projector(args.ssl_model.projector, output_dim, args.ssl_model.projector_dim)  # will be replaced
    logger.info(f"Restoring best model checkpoint - {trainer.checkpoint_callback.best_model_path}")
    model = simclr.SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                               encoder=encoder, projector=projector)
    lr = LinearEvaluationAccuracy(model=model,
                                  data=build_plain_dataset(args),
                                  device=device,
                                  train_batch_size=args.le_model.train_batch_size,
                                  val_batch_size=args.le_model.val_batch_size,
                                  n_cpus=args.n_cpus,
                                  base_lr=args.le_model.optimizer.base_lr,
                                  weight_decay=args.le_model.optimizer.weight_decay,
                                  momentum=args.le_model.optimizer.momentum,
                                  epochs=args.le_model.num_epochs,
                                  val_interval=args.le_val_interval,
                                  output_dir=os.path.join(args.output_dir, "SimCLR_Linear_Evaluation_Callback"),
                                  progressbar=is_running_on_slurm() is False,
                                  )
    acc = lr.compute_accuracy('test')
    lr.save_features_and_model_state_dict(path=trainer.log_dir)
    logger.info(f"Final linear evaluation test accuracy: {acc}")


def build_plain_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10Plain(args.dataset_path)
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
