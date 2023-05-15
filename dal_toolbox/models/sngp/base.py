import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning as L

from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from ..utils.random_features import RandomFeatureGaussianProcess


class SNGPModel(L.LightningModule):
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

    def on_train_epoch_start(self):
        self._reset_precision_matrix()

    def on_train_epoch_end(self):
        self._synchronize_precision_matrix()

    def _reset_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.reset_precision_matrix()

    def _synchronize_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.synchronize_precision_matrix()

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
        inputs, targets = batch
        logits = self(inputs, mean_field=True)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def _gather(self, val):
        if not dist.is_available() or not dist.is_initialized():
            return val
        gathered_val = self.all_gather(val)
        val = torch.cat([v for v in gathered_val])
        return val

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=.9, weight_decay=0.01)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
        else:
            optimizer_params = {} if self.optimizer_params is None else self.optimizer_params
            optimizer = self.optimizer(self.parameters(), **optimizer_params)

        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler_params = {} if self.lr_scheduler_params is None else self.lr_scheduler_params
        lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def log_marg_likelihood(self, inputs, targets, prior_precision=2000):
        # N, D = len(inputs)
        precision_matrix = self.last.precision_matrix
        prior_precision_matrix = prior_precision*torch.eye(len(self.last.precision_matrix))
        weight_map = self.last.beta.weight.data

        logits = self(inputs)
        log_likelihood = -F.cross_entropy(logits, targets, reduction='sum')
        complexity_term = 0.5 * (
            torch.logdet(precision_matrix) - torch.logdet(prior_precision_matrix) +
            weight_map@prior_precision_matrix@weight_map.T
        )
        lml = log_likelihood - complexity_term
        return lml.sum()

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
