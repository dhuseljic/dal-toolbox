import abc
import copy
import inspect
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as L

from collections import defaultdict
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
            scheduler_interval='epoch'
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_interval = scheduler_interval
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

        # TODO(dhuseljic): not working with functools
        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict()) if optimizer else None
        self.init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict()) if lr_scheduler else None

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
            # TODO(dhuseljic): when not in cpu, batchnorm running mean is in inference mode..
            self.model.cpu()
            self.model.load_state_dict(self.init_model_state)
        self.optimizer.load_state_dict(self.init_optimizer_state)
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.init_scheduler_state)
        self.configure_optimizers()

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
            return optimizer
        if isinstance(self.optimizer, functools.partial):
            self.optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return self.optimizer
        if isinstance(self.lr_scheduler, functools.partial):
            self.lr_scheduler = self.lr_scheduler(self.optimizer)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'interval': self.scheduler_interval
            }
        }

    def on_train_start(self):
        self.train()

    def on_train_end(self):
        self.eval()

    def log_train_metrics(self, logits, targets):
        if self.train_metrics is not None:
            metrics = {metric_name: metric(logits, targets)
                       for metric_name, metric in self.train_metrics.items()}
            self.log_dict(self.train_metrics, prog_bar=True)

    def log_val_metrics(self, logits, targets):
        if self.val_metrics is not None:
            metrics = {metric_name: metric(logits, targets)
                       for metric_name, metric in self.val_metrics.items()}
            self.log_dict(self.val_metrics, prog_bar=True)

    @torch.no_grad()
    def get_model_outputs(self, dataloader, output_types: list, device: str = 'cpu', **kwargs):
        self.eval()
        self.to(device)

        outputs = defaultdict(list)
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1]

            features = self.model.forward_features(inputs)
            logits = self.model.forward_head(features)

            for output_type in output_types:
                if output_type == 'features':
                    if features.dim() == 4: # CNN Style
                        pooled_features = self.model.forward_head(features, pre_logits=True)
                    elif features.dim() == 3:  # ViT Style
                        pooled_features = self.model.forward_head(features, pre_logits=True)
                    else:
                        pooled_features = features
                    outputs['features'].append(pooled_features.cpu())
                elif output_type == 'logits':
                    outputs['logits'].append(logits.cpu())
                elif output_type == 'labels':
                    outputs['labels'].append(labels)
                elif output_type == 'mean_field_logits':
                    logits_mean_field = self.model.forward_head(features, mean_field=True)
                    outputs['mean_field_logits'].append(logits_mean_field.cpu())
                elif output_type == 'monte_carlo_logits':
                    raise NotImplementedError('Output type `monte_carlo_logits` is not implemented yet.')
                else:
                    raise NotImplementedError()
        outputs = {key: torch.cat(val) if isinstance(
            val[0], torch.Tensor) else val for key, val in outputs.items()}
        return outputs
