import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.base import BaseModule
from ..utils.mixup import mixup


class DeterministicModel(BaseModule):

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True)

        if self.train_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.train_metrics.items()}
            self.log_dict(self.train_metrics, prog_bar=True)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        if self.val_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.val_metrics.items()}
            self.log_dict(self.val_metrics, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        logits = self.model(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets


class DeterministicLabelsmoothingModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            label_smoothing: float,
            optimizer: torch.optim.Optimizer = None,
            # optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            # lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, optimizer, lr_scheduler, train_metrics, val_metrics)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)


class DeterministicMixupModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            mixup_alpha: float,
            optimizer: torch.optim.Optimizer = None,
            # optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            # lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, optimizer, lr_scheduler, train_metrics, val_metrics)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

    def training_step(self, batch):
        inputs, targets = batch
        targets_one_hot = F.one_hot(targets, self.num_classes)
        batch_mixup = mixup(inputs, targets_one_hot, self.mixup_alpha)
        return super().training_step(batch_mixup)
