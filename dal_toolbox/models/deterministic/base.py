from typing import Any
import torch
import warnings
import lightning as L
import torch.nn.functional as F

from dal_toolbox.metrics.generalization import Accuracy


class DeterministicModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def training_step(self, batch):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        self.train_accuracy(logits, targets)

        train_stats = {'train_loss': loss, 'train_acc': self.train_accuracy}
        self.log_dict(train_stats, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO(dhuseljic): how to handle?
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)

        self.val_accuracy(logits, targets)

        val_stats = {'val_loss': loss, 'val_acc': self.val_accuracy}
        self.log_dict(val_stats, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)

        self.test_accuracy(logits, targets)

        val_stats = {'test_loss': loss, 'test_acc': self.test_accuracy}
        self.log_dict(val_stats, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO(dhuseljic): discuss with mherde; maybe good for AL
        inputs, targets = batch
        logits = self(inputs)
        return logits, targets

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, momentum=.9, weight_decay=0.01)
        warnings.warn(f'Using default optimizer: {optimizer}.')
        return optimizer
