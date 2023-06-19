import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.base import BaseModule
from ..utils.mixup import mixup
from dal_toolbox import metrics

class DeterministicAGLAEModel(BaseModule):

    def training_step(self, batch):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        self.log('train_loss', loss, prog_bar=True)
        self.log_train_metrics(logits, batch['labels'])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        self.log_val_metrics(logits, batch['labels'])
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch['input_ids'], batch['attention_mask'])
        logits = self._gather(logits)
        targets = self._gather(batch['labels'])
        #cc_fn = metrics.Accuracy().to('cuda')
        #self.log('test_accuracy', acc_fn(logits, targets), prog_bar=True)
        return logits, targets 
    

    
class DeterministicModel(BaseModule):

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(targets))

        self.log_train_metrics(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True, batch_size=len(targets))

        self.log_val_metrics(logits, targets)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        logits = self.model(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets


class DeterministicMixupModel(DeterministicModel):

    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            mixup_alpha: float,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

    def training_step(self, batch):
        inputs, targets = batch
        targets_one_hot = F.one_hot(targets, self.num_classes)
        batch_mixup = mixup(inputs, targets_one_hot, self.mixup_alpha)
        return super().training_step(batch_mixup)
