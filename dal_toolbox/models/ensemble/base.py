
import torch
import torch.nn as nn
import lightning as L

from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from ..utils.base import BaseModule
from ...metrics import GibbsCrossEntropy


class EnsembleModel(BaseModule):

    def __init__(
            self,
            member_list: list,
            optimizer_list: list = None,
            lr_scheduler_list: list = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=GibbsCrossEntropy()
    ):
        super().__init__(None, None, None, train_metrics, val_metrics, loss_fn)
        self.members = nn.ModuleList(member_list)
        self.optimizer_list = optimizer_list
        self.lr_scheduler_list = lr_scheduler_list

        self.automatic_optimization = False
        self.val_loss_fn = val_loss_fn

    def forward(self, x):
        logits_list = []
        for member in self.members:
            logits_list.append(member(x))
        return torch.stack(logits_list, dim=1)

    def training_step(self, batch):
        inputs, targets = batch
        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        for i, (member, optimizer, lr_scheduler) in enumerate(zip(self.members, optimizers, lr_schedulers)):
            logits = member(inputs)
            loss = self.loss_fn(logits, targets)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.log(f'loss_member{i}', loss, prog_bar=True)
            if self.trainer.is_last_batch:
                lr_scheduler.step()

            # TODO: How to handle metrics

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.val_loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        if self.val_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.val_metrics.items()}
            self.log_dict(self.val_metrics, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch[0]
        targets = batch[1]
        logits = self(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def configure_optimizers(self):
        if self.optimizer_list is None:
            optimizers = []
            for member in self.members:
                optimizer = torch.optim.SGD(member.parameters(), lr=1e-1, momentum=.9)
                optimizers.append(optimizer)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
            return optimizers

        if self.lr_scheduler_list is None:
            return self.optimizer_list

        return self.optimizer_list, self.lr_scheduler_list
