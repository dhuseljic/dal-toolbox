import abc
import copy
import functools

import torch
import torch.nn as nn
import torch.distributed as dist

import lightning as L

from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class BaseModule(L.LightningModule, abc.ABC):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

        # TODO(dhuseljic): not working with functools
        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _gather(self, val):
        if not dist.is_available() or not dist.is_initialized():
            return val
        gathered_val = self.all_gather(val)
        val = torch.cat([v for v in gathered_val])
        return val

    def reset_states(self, reset_model_parameters=True):
        if reset_model_parameters:
            self.model.load_state_dict(self.init_model_state)
        self.optimizer.load_state_dict(self.init_optimizer_state)
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.init_scheduler_state)
        self.configure_optimizers()

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, momentum=.9, weight_decay=0.01)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
            return optimizer
        if isinstance(self.optimizer, functools.partial):
            self.optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return self.optimizer
        if isinstance(self.lr_scheduler, functools.partial):
            self.lr_scheduler = self.lr_scheduler(self.optimizer)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}

    def log_train_metrics(self, logits, targets):
        if self.train_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.train_metrics.items()}
            self.log_dict(self.train_metrics, prog_bar=True)

    def log_val_metrics(self, logits, targets):
        if self.val_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.val_metrics.items()}
            self.log_dict(self.val_metrics, prog_bar=True)

    # TODO(dhuseljic): Discuss
    @torch.inference_mode()
    def get_logits(self, *args, **kwargs):
        kwargs['device'] = self.device
        if not hasattr(self.model, 'get_logits'):
            raise NotImplementedError('The `get_logits` method is not implemented.')
        return self.model.get_logits(*args, **kwargs)

    @torch.inference_mode()
    def get_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_representations'):
            raise NotImplementedError('The `get_representations` method is not implemented.')
        return self.model.get_representations(*args, **kwargs)

    @torch.inference_mode()
    def get_grad_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_grad_representations'):
            raise NotImplementedError('The `get_grad_representations` method is not implemented.')
        return self.model.get_grad_representations(*args, **kwargs)
