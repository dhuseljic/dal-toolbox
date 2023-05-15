import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import lightning as L

from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from ..utils.mixup import mixup


class DeterministicModel(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer = None,
            optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

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
        # TODO(dhuseljic): discuss with mherde; maybe can be used for AL?
        inputs = batch[0]
        targets = batch[1]
        indices = batch[2] if len(batch) == 3 else torch.zeros_like(targets).long()

        logits = self(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        indices = self._gather(indices)
        return logits, targets, indices

    def _gather(self, val):
        if not dist.is_available() or not dist.is_initialized():
            return val
        gathered_val = self.all_gather(val)
        val = torch.cat([v for v in gathered_val])
        return val

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, momentum=.9, weight_decay=0.01)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
        else:
            optimizer_params = {} if self.optimizer_params is None else self.optimizer_params
            optimizer = self.optimizer(self.parameters(), **optimizer_params)

        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler_params = {} if self.lr_scheduler_params is None else self.lr_scheduler_params
        lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    # TODO(dhuseljic): Discuss
    def get_logits(self, *args, **kwargs):
        kwargs['device'] = self.device
        if not hasattr(self.model, 'get_logits'):
            raise NotImplementedError('The `get_logits` method is not implemented.')
        return self.model.get_logits(*args, **kwargs)

    def get_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_representations'):
            raise NotImplementedError('The `get_representations` method is not implemented.')
        return self.model.get_representations(*args, **kwargs)

    def get_grad_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_grad_representations'):
            raise NotImplementedError('The `get_grad_representations` method is not implemented.')
        return self.model.get_grad_representations(*args, **kwargs)


class DeterministicLabelsmoothingModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            label_smoothing: float,
            optimizer: torch.optim.Optimizer = None,
            optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, train_metrics, val_metrics)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)


class DeterministicMixupModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            mixup_alpha: float,
            optimizer: torch.optim.Optimizer = None,
            optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, train_metrics, val_metrics)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

    def training_step(self, batch):
        inputs, targets = batch
        targets_one_hot = F.one_hot(targets, self.num_classes)
        batch_mixup = mixup(inputs, targets_one_hot, self.mixup_alpha)
        return super().training_step(batch_mixup)
