import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import lightning as L


class MCDropoutModel(L.LightningModule):
    def __init__(self, model, metrics=None):
        super().__init__()
        self.model = model
        self.metrics = nn.ModuleDict(metrics)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def mc_forward(self, *args, **kwargs):
        return self.model.mc_forward(*args, **kwargs)

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True)

        if self.metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.metrics.items()}
            self.log_dict(self.metrics, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO(dhuseljic): discuss with mherde; maybe can be used for AL?
        inputs, targets = batch
        logits = self.model.mc_forward(inputs)

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
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, momentum=.9)
        warnings.warn(f'Using default optimizer: {optimizer}.')
        return optimizer
