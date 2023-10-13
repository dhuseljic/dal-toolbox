import json
import logging
import math
import os
import sys

import hydra
import lightning as L
import numpy as np
import torch.cuda
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from pytorch_optimizer import LARS
from torch import nn
from torch.utils.data import DataLoader

from dal_toolbox import datasets, metrics
from dal_toolbox.datasets.utils import FeatureDataset
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss, LinearEvaluationAccuracy
from dal_toolbox.models.utils.callbacks import MetricHistory
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
from dal_toolbox.utils import is_running_on_slurm, seed_everything


def build_ssl(name, args):
    if name == 'simclr':
        return build_simclr(args)
    else:
        raise NotImplementedError(f"Self-supervised model \"{name}\" is not implemented!")


def build_encoder(name, return_output_dim=False, dataset=""):
    if name == 'resnet18_deterministic':
        encoder = deterministic.resnet.ResNet18(num_classes=1, imagenethead=("ImageNet" in dataset))
        encoder.linear = nn.Identity()  # Replace layer after max pool

        encoder_output_dim = 512
    elif name == "resnet50_deterministic":
        encoder = deterministic.resnet.ResNet50(num_classes=1, imagenethead=("ImageNet" in dataset))
        encoder.linear = nn.Identity()

        encoder_output_dim = 2048
    elif name == "wide_resnet_28_10":
        encoder = deterministic.wide_resnet.wide_resnet_28_10(num_classes=1, dropout_rate=0.3)
        encoder.linear = nn.Identity()

        encoder_output_dim = 640
    else:
        raise NotImplementedError(f"Encoder model \"{name}\" is not implemented!")

    if return_output_dim:
        return encoder, encoder_output_dim
    return encoder


def build_projector(name, input_dim, output_dim):
    if name == 'mlp':
        projector = nn.Sequential(nn.Linear(input_dim, output_dim),
                                  nn.ReLU(),
                                  nn.Linear(output_dim, output_dim))
    else:
        raise NotImplementedError(f"Projector mdoel \"{name}\" is not implemented!")
    return projector


def build_simclr(args) -> (nn.Module, nn.Module):
    encoder, encoder_output_dim = build_encoder(args.ssl_model.encoder, True, args.dataset.name)
    projector = build_projector(args.ssl_model.projector, encoder_output_dim, args.ssl_model.projector_dim)

    optimizer = LARS(
        params=list(encoder.parameters()) + list(projector.parameters()),
        lr=args.ssl_model.optimizer.base_lr * math.sqrt(
            args.ssl_model.train_batch_size * args.ssl_model.accumulate_batches),
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
    def __init__(self, device, args, dataset, logger=None):
        self.args = args
        self.device = device
        self.interval = args.le_model.callback.interval
        self.dataset = dataset
        self.logger = logger
        self.metrics = {"interval": self.interval, "final_train_acc": [], "final_val_acc": [], "final_test_acc": []}

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.interval == 0:
            if self.logger is not None:
                self.logger.info("Starting linear evaluation callback.")
            lr = LinearEvaluationAccuracy(model=pl_module,
                                          data=self.dataset,
                                          device=self.device,
                                          train_batch_size=self.args.le_model.train_batch_size,
                                          val_batch_size=self.args.le_model.val_batch_size,
                                          n_cpus=self.args.n_cpus,
                                          base_lr=self.args.le_model.optimizer.base_lr,
                                          weight_decay=self.args.le_model.optimizer.weight_decay,
                                          momentum=self.args.le_model.optimizer.momentum,
                                          epochs=self.args.le_model.num_epochs,
                                          output_dir=os.path.join(
                                              self.args.output_dir, "lightning", "SimCLR_Linear_Evaluation_Callback"),
                                          progressbar=is_running_on_slurm() is False,
                                          )
            train_acc = lr.compute_accuracy()
            val_acc = lr.compute_accuracy('val')
            test_acc = lr.compute_accuracy('test')

            self.metrics["final_train_acc"].append(train_acc)
            self.metrics["final_val_acc"].append(val_acc)
            self.metrics["final_test_acc"].append(test_acc)

            if self.logger is not None:
                self.logger.info(f"Epoch {trainer.current_epoch} - "
                                 f"Linear evaluation accuracy: "
                                 f"Train : {train_acc:3f}, "
                                 f"Validation: {val_acc:3f}, "
                                 f"Test: {test_acc:3f}")


@hydra.main(version_base=None, config_path="../self_supervised_learning/config", config_name="config")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # To be reproducible
    seed_everything(args.random_seed)

    # Create a Data Module
    data = build_contrastive_dataset(args)
    plain_data = build_plain_dataset(args)
    train_dataloader = DataLoader(data.train_dataset,
                                  batch_size=args.ssl_model.train_batch_size,
                                  num_workers=args.n_cpus,
                                  shuffle=True)

    val_dataloader = DataLoader(data.val_dataset,
                                batch_size=args.ssl_model.val_batch_size,
                                num_workers=args.n_cpus,
                                shuffle=False)

    model = build_ssl(args.ssl_model.name, args)
    logger.info(f"Using square-root adjusted learning rate "
                f"{math.sqrt(args.ssl_model.train_batch_size * args.ssl_model.accumulate_batches) * args.ssl_model.optimizer.base_lr}")
    if args.ssl_model.accumulate_batches > 1:
        logger.info(f"Effective batch size is {args.ssl_model.accumulate_batches * args.ssl_model.train_batch_size}")

    callbacks = [
        ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc_top5", every_n_epochs=1),
        LearningRateMonitor("epoch"),
        MetricHistory()
    ]

    if args.le_model.callback.enabled:
        callbacks.append(LinearEvaluationCallback(device, args, plain_data, logger))

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    # Create a Trainer Module
    trainer = L.Trainer(
        default_root_dir=os.path.join(args.output_dir, "lightning", "SimCLR"),
        accumulate_grad_batches=args.ssl_model.accumulate_batches,
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
                                  data=plain_data,
                                  device=device,
                                  train_batch_size=args.le_model.train_batch_size,
                                  val_batch_size=args.le_model.val_batch_size,
                                  n_cpus=args.n_cpus,
                                  base_lr=args.le_model.optimizer.base_lr,
                                  weight_decay=args.le_model.optimizer.weight_decay,
                                  momentum=args.le_model.optimizer.momentum,
                                  epochs=args.le_model.num_epochs,
                                  output_dir=os.path.join(
                                      args.output_dir, "lightning", "SimCLR_Linear_Evaluation_Callback"),
                                  progressbar=is_running_on_slurm() is False,
                                  )

    final_train_acc = lr.compute_accuracy('train')
    final_val_acc = lr.compute_accuracy('val')
    final_test_acc = lr.compute_accuracy('test')
    logger.info(f"Epoch {trainer.current_epoch} - "
                f"Linear evaluation accuracy: "
                f"Train : {final_train_acc:3f}, "
                f"Validation: {final_val_acc:3f}, "
                f"Test: {final_test_acc:3f}")

    results = {}
    simclr_metrics: list = callbacks[2].metrics
    simclr_metrics_reordered = {"train_loss": [], "train_acc_top1": [], "train_acc_top5": [],
                                "val_loss": [], "val_acc_top1": [], "val_acc_top5": []}

    for epoch in simclr_metrics:
        simclr_metrics_reordered["train_loss"].append(epoch["train_loss"])
        simclr_metrics_reordered["train_acc_top1"].append(epoch["train_acc_top1"])
        simclr_metrics_reordered["train_acc_top5"].append(epoch["train_acc_top5"])
        if "val_loss" in epoch.keys():
            simclr_metrics_reordered["val_loss"].append(epoch["val_loss"])
            simclr_metrics_reordered["val_acc_top1"].append(epoch["val_acc_top1"])
            simclr_metrics_reordered["val_acc_top5"].append(epoch["val_acc_top5"])
        else:
            simclr_metrics_reordered["val_loss"].append(np.nan)
            simclr_metrics_reordered["val_acc_top1"].append(np.nan)
            simclr_metrics_reordered["val_acc_top5"].append(np.nan)

    results["SimCLR"] = simclr_metrics_reordered
    results["LinearEvaluation"] = callbacks[3].metrics
    results["FinalLinearEvaluation"] = {"final_train_acc": final_train_acc,
                                        "final_val_acc": final_val_acc,
                                        "final_test_acc": final_test_acc}

    file_name = os.path.join(args.output_dir, 'results.json')
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, sort_keys=False)
    logger.info(f"Saved results under {file_name}")

    file_name = save_feature_dataset_and_model(model, plain_data, device, path=args.output_dir,
                                               name=f"{args.ssl_model.encoder}_{args.dataset.name}_{final_val_acc:.3f}")
    logger.info(f"Saved model under {file_name}")


def build_plain_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10Plain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100Plain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHNPlain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'ImageNet':
        data = datasets.ImageNetPlain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'ImageNet50':
        data = datasets.ImageNet50Plain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'ImageNet100':
        data = datasets.ImageNet100Plain(args.dataset_path, seed=args.random_seed)
    elif args.dataset.name == 'ImageNet200':
        data = datasets.ImageNet200Plain(args.dataset_path, seed=args.random_seed)
    else:
        sys.exit(f"Dataset {args.dataset.name} not implemented.")

    return data


def build_contrastive_dataset(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10Contrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                           seed=args.random_seed)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100Contrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                            seed=args.random_seed)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHNContrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                        r_prob=args.dataset.rotation_probability, seed=args.random_seed)
    elif args.dataset.name == 'ImageNet':
        data = datasets.ImageNetContrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                            seed=args.random_seed)
    elif args.dataset.name == 'ImageNet50':
        data = datasets.ImageNet50Contrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                              seed=args.random_seed)
    elif args.dataset.name == 'ImageNet100':
        data = datasets.ImageNet100Contrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                               seed=args.random_seed)
    elif args.dataset.name == 'ImageNet200':
        data = datasets.ImageNet200Contrastive(args.dataset_path, cds=args.dataset.color_distortion_strength,
                                               seed=args.random_seed)
    else:
        sys.exit(f"Dataset {args.dataset.name} not implemented.")

    return data


def save_feature_dataset_and_model(model, dataset, device, path, name):
    trainset = FeatureDataset(model, dataset.full_train_dataset, device)
    testset = FeatureDataset(model, dataset.test_dataset, device)

    path = os.path.join(path + f"{name}.pth")
    torch.save({'trainset': trainset,
                'testset': testset,
                'model': model.encoder.state_dict()}, path)
    return path


if __name__ == '__main__':
    main()
