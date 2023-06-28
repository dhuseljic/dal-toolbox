import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import DeterministicModel


def freeze_bn(model):
    """Freezes the existing batch_norm layers in a module.

    Args:
        model (nn.Module): Deep neural network with batch norm layers.

    Returns:
        dict: Returns a dictionary of the tracked batchnorm statistics such as `running_mean` or `running_var`.
    """
    backup = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.SyncBatchNorm) or isinstance(module, nn.BatchNorm2d):
            backup[name + '.running_mean'] = module.running_mean.data.clone()
            backup[name + '.running_var'] = module.running_var.data.clone()
            backup[name + '.num_batches_tracked'] = module.num_batches_tracked.data.clone()
    return backup


def unfreeze_bn(model, backup):
    for name, module in model.named_modules():
        if isinstance(module, nn.SyncBatchNorm) or isinstance(module, nn.BatchNorm2d):
            module.running_mean.data = backup[name + '.running_mean']
            module.running_var.data = backup[name + '.running_var']
            module.num_batches_tracked.data = backup[name + '.num_batches_tracked']


class DeterministicPseudoLabelModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            cutoff_proba: float = .95,
            unsup_warmup: float = .4,
            unsup_weight: float = 1.,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics, scheduler_interval='step')
        self.cutoff_proba = cutoff_proba
        self.unsup_warmup = unsup_warmup
        self.unsup_weight = unsup_weight

    def training_step(self, batch):
        labeled_batch = batch['labeled']
        inputs, targets = labeled_batch[0], labeled_batch[1]

        logits = self(inputs)
        supervised_loss = self.loss_fn(logits, targets)

        unlabeled_batch = batch['unlabeled']
        unlabeled_inputs = unlabeled_batch[0]

        bn_backup = freeze_bn(self.model)
        unlabeled_logits = self(unlabeled_inputs)
        unfreeze_bn(self.model, bn_backup)
        unlabeled_probas = torch.softmax(unlabeled_logits.detach(), dim=-1)
        max_probas, pseudo_label = torch.max(unlabeled_probas, dim=-1)
        mask = max_probas.ge(self.cutoff_proba)
        unsupervised_loss = torch.mean(F.cross_entropy(unlabeled_logits, pseudo_label, reduction='none') * mask)

        warmup_factor = np.clip(self.global_step / (self.unsup_warmup*self.trainer.max_steps), a_min=0, a_max=1)
        loss = supervised_loss + warmup_factor * self.unsup_weight * unsupervised_loss

        self.log('warmup_factor', warmup_factor, prog_bar=True)
        self.log('supervised_loss', supervised_loss, prog_bar=True)
        self.log('unsupervised_loss', unsupervised_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log_train_metrics(logits, targets)
        return loss


class DeterministicPiModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            unsup_warmup: float = .4,
            unsup_weight: float = 1.,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics, scheduler_interval='step')
        self.num_classes = num_classes
        self.unsup_warmup = unsup_warmup
        self.unsup_weight = unsup_weight

    def training_step(self, batch):
        labeled_batch = batch['labeled']
        inputs, targets = labeled_batch[0], labeled_batch[1]

        logits = self(inputs)
        supervised_loss = self.loss_fn(logits, targets)

        unlabeled_batch = batch['unlabeled'][0]
        unlabeled_inputs_aug1 = unlabeled_batch[0]
        unlabeled_inputs_aug2 = unlabeled_batch[1]

        bn_backup = freeze_bn(self.model)
        unlabeled_logits_aug1 = self.model(unlabeled_inputs_aug1)
        unlabeled_logits_aug2 = self.model(unlabeled_inputs_aug2)
        unfreeze_bn(self.model, bn_backup)

        unlabeled_probas_aug1 = torch.softmax(unlabeled_logits_aug1, dim=-1)
        unlabeled_probas_aug2 = torch.softmax(unlabeled_logits_aug2.detach(), dim=-1)
        unsupervised_loss = F.mse_loss(unlabeled_probas_aug1, unlabeled_probas_aug2)

        warmup_factor = np.clip(self.global_step / (self.unsup_warmup*self.trainer.max_steps), a_min=0, a_max=1)
        loss = supervised_loss + warmup_factor * self.unsup_weight * unsupervised_loss

        self.log('warmup_factor', warmup_factor, prog_bar=True)
        self.log('supervised_loss', supervised_loss, prog_bar=True)
        self.log('unsupervised_loss', unsupervised_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log_train_metrics(logits, targets)
        return loss


class DeterministicFixMatchModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            cutoff_proba: float = .95,
            unsup_weight: float = 1.,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics, scheduler_interval='step')
        self.cutoff_proba = cutoff_proba
        self.unsup_weight = unsup_weight

    def training_step(self, batch):
        labeled_batch = batch['labeled']
        inputs, targets = labeled_batch[0], labeled_batch[1]

        # Supervised
        logits = self(inputs)
        supervised_loss = self.loss_fn(logits, targets)

        # Unsupervised
        unlabeled_batch = batch['unlabeled'][0]
        inputs_weak = unlabeled_batch[0]
        inputs_strong = unlabeled_batch[1]

        logits_strong = self(inputs_strong)
        with torch.no_grad():
            logits_weak = self(inputs_weak)

        # TODO: dist align missing
        probas_weak = torch.softmax(logits_weak, dim=-1)
        max_probas_weak, pseudo_labels_weak = torch.max(probas_weak, dim=-1)
        mask = max_probas_weak.ge(self.cutoff_proba)
        unsupervised_loss = mask * F.cross_entropy(logits_strong, pseudo_labels_weak, reduction='none')
        unsupervised_loss = torch.mean(unsupervised_loss)

        # Total Loss
        loss = supervised_loss + self.unsup_weight * unsupervised_loss

        self.log('supervised_loss', supervised_loss, prog_bar=True)
        self.log('unsupervised_loss', unsupervised_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log_train_metrics(logits, targets)
        return loss
