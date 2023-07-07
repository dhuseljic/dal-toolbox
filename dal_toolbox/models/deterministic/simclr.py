import logging
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .base import DeterministicModel
from .. import deterministic
from ..utils.base import BaseModule
from ... import metrics
from ...datasets.base import BaseData
from ...datasets.utils import FeatureDataset


# TODO (ynagel) This should probably be moved to a "losses.py" file of some sorts
class InfoNCELoss(nn.Module):
    """
    The InfoNCE contrastive loss.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Initializes ``InfoNCELoss``.

        Args:
            temperature: The temperature.
        """
        super().__init__()

        assert temperature > 0.0, "The temperature must be a positive float!"
        self.temperature = temperature

    def forward(self, batch, targets):
        """
        Returns: The InfoNCE loss of the batch.

        Args:
            batch: The contrastive batch.
            targets: For compatibility and are ignored.
        """
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(batch[:, None, :], batch[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        infoNCE = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        infoNCE = infoNCE.mean()

        return infoNCE


class LinearEvaluationAccuracy:
    """
    Wrapper to calculate the linear evaluation accuracy.

    The linear evaluation accuracy is the accuracy of a linear classifier trained on the features of another network
    (e.g. a self-supervision model). This wrapper extracts the encoder of the network ``model`` and the features from
    the dataset ``data`` passed to ``__init__(model, data)``. The linear classifier consists of a single
    ``nn.Linear(feature_dimension, num_classes)`` layer, which is trained with the SGD optimizer.
    """

    def __init__(self,
                 model: BaseModule,
                 data: BaseData,
                 device: torch.device,
                 train_batch_size: int = 4096,
                 val_batch_size: int = 512,
                 n_cpus: int = 1,
                 base_lr: float = 0.1,
                 use_linear_lr_scaling: bool = True,
                 weight_decay: float = 0.0,
                 momentum: float = 0.9,
                 epochs: int = 90,
                 val_interval: int = 25,
                 output_dir: str = "",
                 checkpoint: bool = False,
                 progressbar: bool = False) -> None:
        """
        Initializes ``LinearEvaluationAccuracy``. This incorporates training the model.

        Args:
            model: The model from which the features are to be extracted from.
            data: The dataset used to extract the features from ``model``.
            device: The ``torch.device`` on which the feature representations are calculated.
            train_batch_size: The batch size used for training.
            val_batch_size: The batch size used for training.
            n_cpus: Number of cpu processes used for the dataloaders.
            base_lr: The base learning rate used for the linear classifier.
            use_linear_lr_scaling: Whether base_lr should be linearly scaled based on the batch size. This is referred
                to as `linear learning rate scaling` and the learning rate is calculated in the following way:
                ``base_lr * train_batch_size / 256``
            weight_decay: The weight decay of the SGD optimizer
            momentum: The momentum of the SGD optimizer.
            epochs: The number of epochs the model is trained for.
            val_interval: In which interval the trainer checks performance on the validation dataset.
            output_dir: The directory in which logs and checkpoints are saved.
            checkpoint: Whether model checkpoints should be saved during training.
            progressbar: Whether the pytorch lightning trainer should display a progress bar.
        """
        self.checkpoint = checkpoint
        self.encoder = model.encoder

        # From my testing this is slightly faster than passing the normal dataset through the frozen backbone in each
        # epoch, however only when training for more than ~15 epochs. Also, the dataset has to be "plain", meaning no
        # transforms take place
        self.trainset = FeatureDataset(self.encoder, data.train_dataset, device)
        self.valset = FeatureDataset(self.encoder, data.val_dataset, device)
        self.testset = FeatureDataset(self.encoder, data.test_dataset, device)

        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=train_batch_size,
                                           num_workers=n_cpus,
                                           shuffle=True)

        self.val_dataloader = DataLoader(self.valset,
                                         batch_size=val_batch_size,
                                         num_workers=n_cpus,
                                         shuffle=False)

        self.test_dataloader = DataLoader(self.testset,
                                          batch_size=val_batch_size,
                                          num_workers=n_cpus,
                                          shuffle=False)

        if hasattr(data, "num_classes"):
            num_classes = data.num_classes
        else:  # Manually calculate number of classes
            num_classes = len(torch.unique([label for _, label in self.trainset]))

        output_dim = self.trainset.features.shape[1]
        model = nn.Linear(output_dim, num_classes)

        if use_linear_lr_scaling:
            lr = base_lr * train_batch_size / 256
        else:
            lr = base_lr

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            default_root_dir=output_dir,
            accelerator="auto",
            max_epochs=epochs,
            enable_checkpointing=checkpoint,
            callbacks=callbacks,
            check_val_every_n_epoch=val_interval,
            enable_progress_bar=progressbar,
            num_sanity_val_steps=0
        )

        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

    def compute_accuracy(self, dataset: str = 'train') -> float:
        """
        Returns either train, validation or test accuracy, depending on ``dataset``.

        The train and validation accuracy are taken from the last epoch, the test accuracy is newly computed. When
        checkpointing is enabled, the test accuracy is computed on the best model saved based on the validation accuracy.
        Args:
            dataset: Which dataset to compute accuracy on. Can be either ``train``, ``val`` or ``test``.
        """
        if dataset == 'train':
            return self.trainer.logged_metrics["train_acc"].float()
        elif dataset == 'val':
            return self.trainer.logged_metrics["val_acc"].float()
        elif dataset == 'test':
            predictions = self.trainer.predict(self.model, self.test_dataloader,
                                               ckpt_path='best' if self.checkpoint else None)

            logits = torch.cat([pred[0] for pred in predictions])
            targets = torch.cat([pred[1] for pred in predictions])
            return metrics.Accuracy()(logits, targets).item()
        else:
            raise NotImplementedError(f"Data split {dataset} is not implemented for compute().")

    def save_features_and_model_state_dict(self, name: str = "model_features_dict", path: str = "") -> None:
        """
        Saves the encoder features as well as the feature datasets used during training/testing.
        Args:
            name: The name of file to save the information to.
            path: The path of where to save the file.
        """
        path = os.path.join(path + os.path.sep + f"{name}.pth")
        torch.save({'trainset': self.trainset,
                    'valset': self.valset,
                    'testset': self.testset,
                    'model': self.encoder.state_dict()}, path)


class SimCLR(DeterministicModel):
    """
    Implements `A Simple Framework for Contrastive Learning of Visual Representations` by Chen et al. in a pytorch
    lightning module.

    Attributes:
        encoder: The encoder part of SimCLR.
        projector: The projector part of SimCLR
    """

    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            log_on_epoch_end: bool = True,
            loss_fn: nn.Module = InfoNCELoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ) -> None:
        """
        Initializes ``SimCLR``.

        The ``encoder`` and ``projector`` have to be compatible (i.e. the output dimension of the encoder has to be the
        same as the input dimension of the projector), as they are simply wrapped in a ``nn.Sequential`` module.
        Args:
            encoder: The encoder module.
            projector: The projector module.
            log_on_epoch_end: Whether metrics accumulated during an epoch should be logged at the end of the epoch.
            loss_fn: The loss function. Has to ba a contrastive loss.
            optimizer: The optimizer to train the model with.
            lr_scheduler: The learning rate scheduler to train the model with.
            train_metrics: Training metrics to be calculated during training.
            val_metrics: Validation metrics to be calculated during validation.
        """
        model = nn.Sequential(encoder, projector)

        super().__init__(model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         train_metrics=train_metrics, val_metrics=val_metrics)

        self.encoder = encoder
        self.projector = projector
        self.log_on_epoch_end = log_on_epoch_end

    def training_step(self, batch):
        batch[0] = torch.cat(batch[0], dim=0)
        return super().training_step(batch)

    def validation_step(self, batch, batch_idx):
        batch[0] = torch.cat(batch[0], dim=0)
        super().validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        if self.log_on_epoch_end:
            log_str = "Current Performance-Metric-Values: "
            for metr, val in self.trainer.logged_metrics.items():
                log_str += (metr + " : " + str(round(val.item(), 5)) + ", ")
            logging.info(log_str)
        return super().on_train_epoch_end()
