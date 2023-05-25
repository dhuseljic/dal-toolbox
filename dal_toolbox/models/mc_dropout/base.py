import torch
import torch.nn as nn
from ..utils.base import BaseModule

from ..utils.mcdropout import MCDropoutModule
from ...metrics import GibbsCrossEntropy, ensemble_log_softmax


class MCDropoutModel(BaseModule):
    def __init__(
            self,
            model: MCDropoutModule,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            val_loss_fn=GibbsCrossEntropy(),
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics)
        self.val_loss_fn = val_loss_fn

    def mc_forward(self, *args, **kwargs):
        return self.model.mc_forward(*args, **kwargs)

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)

        self.log('train_loss', loss, prog_bar=True)
        self.log_train_metrics(logits, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        mc_logits = self.mc_forward(inputs)

        loss = self.val_loss_fn(mc_logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        logits = ensemble_log_softmax(mc_logits)
        self.log_val_metrics(logits, targets)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch[0]
        targets = batch[1]
        logits = self.model.mc_forward(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets
