import logging
import math
import os
import sys

import hydra
import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from pytorch_optimizer import LARS
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dal_toolbox import datasets, metrics
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss
from dal_toolbox.models.utils.base import BaseModule
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


class FeatureDataset(Dataset):
    def __init__(self, model, dataset, device):
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)
        features = model.get_representations(dataloader, device)  # TODO This can be speed up by using a single loop
        self.features = features.detach()
        self.labels = [label for _, label in dataset]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LinearEvaluationCallback(Callback):
    def __init__(self, output_dim, args):
        self.args = args
        self.output_dim = output_dim
        self.interval = args.le_model.callback.interval
        self.logger = logging.getLogger(__name__)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.interval == 0:
            self.logger.info("Starting linear evaluation callback.")
            lr = LinearEvaluationAccuracy(pl_module, self.output_dim, self.args)

            self.logger.info(f"Epoch {trainer.current_epoch} - "
                             f"Linear evaluation accuracy: "
                             f"Train : {lr.compute():4f} , Validation: {lr.compute('val'):4f}")


# TODO The structure should be solved differently
class LinearEvaluationAccuracy():
    def __init__(self, model: BaseModule, output_dim: int, args, checkpoint=False):
        data = build_dataset(args)

        # From my testing this is slightly faster than passing the normal dataset through the frozen backbone in each
        # epoch, however only when training for more than ~15 epochs
        self.trainset = FeatureDataset(model.encoder, data.train_dataset, "cuda")  # TODO load device from somewhere
        self.valset = FeatureDataset(model.encoder, data.val_dataset, "cuda")  # TODO load device from somewhere
        self.testset = FeatureDataset(model.encoder, data.test_dataset, "cuda")  # TODO load device from somewhere

        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=args.le_model.train_batch_size,
                                           num_workers=args.n_cpus,
                                           shuffle=True)

        self.val_dataloader = DataLoader(self.valset,
                                         batch_size=args.le_model.val_batch_size,
                                         num_workers=args.n_cpus,
                                         shuffle=False)

        self.test_dataloader = DataLoader(self.testset,
                                          batch_size=args.le_model.val_batch_size,
                                          num_workers=args.n_cpus,
                                          shuffle=False)

        num_classes = data.num_classes
        self.encoder = model.encoder

        model = nn.Linear(output_dim, num_classes)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.ssl_model.optimizer.base_lr * args.le_model.train_batch_size / 256,
                                    weight_decay=args.le_model.optimizer.weight_decay,
                                    momentum=args.le_model.optimizer.momentum)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.le_model.num_epochs)
        self.model = deterministic.DeterministicModel(
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )

        if checkpoint:
            callbacks = [ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc", every_n_epochs=1)]
        else:
            callbacks = []

        self.trainer = L.Trainer(
            default_root_dir=os.path.join(args.output_dir, "SimCLR_Linear_Evaluation_Callback"),
            accelerator="auto",
            max_epochs=args.le_model.num_epochs,
            enable_checkpointing=checkpoint,
            callbacks=callbacks,
            check_val_every_n_epoch=args.le_val_interval,
            enable_progress_bar=is_running_on_slurm() is False,
            num_sanity_val_steps=0
        )

        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
        self.checkpoint = checkpoint

    def compute(self, dataset='train'):
        if dataset == 'train':
            return self.trainer.logged_metrics["train_acc"]
        elif dataset == 'val':
            return self.trainer.logged_metrics["val_acc"]
        elif dataset == 'test':
            predictions = self.trainer.predict(self.model, self.test_dataloader,
                                               ckpt_path='best' if self.checkpoint else None)

            logits = torch.cat([pred[0] for pred in predictions])
            targets = torch.cat([pred[1] for pred in predictions])
            return metrics.Accuracy()(logits, targets).item()
        else:
            sys.exit(f"Data split {dataset} does not exist.")

    def save_features_and_model_state_dict(self, name="model_features_dict", path=""):
        # TODO Is this really the best method?
        path = os.path.join(path + os.path.sep + f"{name}.pth")
        torch.save({'trainset': self.trainset,
                    'valset': self.valset,
                    'testset': self.testset,
                    'model': self.encoder.state_dict()}, path)


@hydra.main(version_base=None, config_path="../self_supervised_learning/config", config_name="config")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))

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
        callbacks.append(LinearEvaluationCallback(model.encoder_output_dim, args))

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
    lr = LinearEvaluationAccuracy(model, output_dim, args, checkpoint=True)
    # TODO This has the side effect of loading the best validation model, which  is currently not clear
    acc = lr.compute('test')
    lr.save_features_and_model_state_dict(path=trainer.log_dir)
    logger.info(f"Final linear evaluation test accuracy: {acc}")


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
