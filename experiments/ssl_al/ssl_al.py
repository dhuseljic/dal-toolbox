import datetime
import json
import logging
import os
import time

import hydra
import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dal_toolbox import datasets, metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.models import deterministic
from dal_toolbox.models.deterministic import resnet
from dal_toolbox.models.deterministic import simclr
from dal_toolbox.models.deterministic.simclr import InfoNCELoss
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import is_running_on_slurm
from experiments.active_learning.active_learning import build_al_strategy


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
    # TODO This currently leaks test data information to the ssl model
    dm = CIFAR10DataModule(
        ds_path=args.dataset_path,
        train_batch_size=args.ssl_model.train_batch_size,
        val_batch_size=args.ssl_model.val_batch_size,
        n_workers=args.n_cpus,
        n_epochs=args.ssl_model.n_epochs,  # TODO What for?
        random_seed=args.random_seed
    )

    # TODO Make flexible
    model = deterministic.resnet.ResNet18(num_classes=4 * args.ssl_model.hidden_dim) # num_classes is the output size of the last linear layer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.ssl_model.optimizer.lr * args.ssl_model.train_batch_size/256,
        weight_decay=args.ssl_model.optimizer.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.ssl_model.n_epochs,
        eta_min=args.ssl_model.optimizer.lr / 50
    )

    # Create a Model Module
    model = simclr.SimCLR(
        model=model,
        loss_fn=InfoNCELoss(args.ssl_model.temperature),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        hidden_dim=args.ssl_model.hidden_dim,
    )

    # Create a Trainer Module
    trainer = L.Trainer(
        default_root_dir=os.path.join(args.cp_path, "SimCLR"),
        accelerator="auto",
        max_epochs=args.ssl_model.n_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_loss", save_top_k=1),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=args.ssl_val_interval,
    )

    logger.info("Training SSL")
    # Train and automatically save top 5 models based on validation accuracy
    trainer.fit(model, datamodule=dm)

    # Setup Query
    logger.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_al_strategy(args)

    # Setup Dataset (Now without contrastive transforms)
    logger.info('Building datasets.')
    data = build_datasets(args)
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

    # TODO Loading the best model
    # model = simclr.SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Splitting off projection head
    model = model.model
    for name, p in model.named_parameters():
        if not "linear" in name:
            p.requires_grad = False # TODO Make adjustable

    model.linear = nn.Linear(512 * model.block.expansion, data.num_classes)
    model.num_classes = data.num_classes

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
            enable_model_summary=False,
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


def build_datasets(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)

    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)

    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.dataset_path)

    return data


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


# TODO Incorporate this somehow into the normal datasets module
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self,
                 ds_path: str = "./data",
                 train_batch_size: int = 256,
                 val_batch_size: int = 128,
                 n_workers: int = 1,
                 n_epochs: int = None,
                 random_seed: int = 42
                 ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.train_ds = self.build_cifar10('train_contrast', self.hparams.ds_path)
        self.val_ds = self.build_cifar10('test_contrast', self.hparams.ds_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.train_batch_size, shuffle=True,
                          num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.val_batch_size, num_workers=self.hparams.n_workers)

    def build_cifar10(self, split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262),
                      return_info=False):
        contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=32),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if split == 'train_contrast':
            ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True,
                                              transform=ContrastiveTransformations(contrast_transforms, n_views=2))
        elif split == 'test_contrast':
            ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True,
                                              transform=ContrastiveTransformations(contrast_transforms, n_views=2))

        if return_info:
            ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
            return ds, ds_info
        return ds


if __name__ == '__main__':
    main()
