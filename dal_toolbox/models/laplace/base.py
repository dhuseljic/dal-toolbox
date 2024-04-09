import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.laplace import LaplaceLayer
from ..utils.base import BaseModule


class LaplaceModel(BaseModule):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            scheduler_interval='epoch',
            predict_kwargs=None,
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics, scheduler_interval)
        self.predict_kwargs = dict(mean_field=True) if predict_kwargs is None else predict_kwargs

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True)

        self.log_train_metrics(logits, targets)
        return loss

    def on_train_epoch_start(self):
        self._reset_precision_matrix()

    def on_train_epoch_end(self):
        self._synchronize_precision_matrix()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        logits = self.model.forward_mean_field(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        self.log_val_metrics(logits, targets)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch

        logits = self.model.forward_mean_field(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def set_mean_field_factor(self, mean_field_factor: float):
        for m in self.model.modules():
            if isinstance(m, LaplaceLayer):
                m.mean_field_factor = mean_field_factor

    def _reset_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, LaplaceLayer):
                m.reset_precision_matrix()

    def _synchronize_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, LaplaceLayer):
                m.synchronize_precision_matrix()

    @torch.no_grad()
    def update_posterior(self, dataloader, lmb=1, gamma=1, likelihood='gaussian'):
        self.eval()

        # check if return_features is in forward_kwargs
        forward_kwargs = inspect.signature(self.model.forward).parameters
        if 'return_features' not in forward_kwargs:
            raise ValueError('Define the kwarg `return_features` in the forward method of your model.')

        # get features
        phis_list = []
        targets_list = []
        for inputs, targets in dataloader:
            logits, phis = self.model(inputs, return_features=True)
            phis_list.append(phis)
            targets_list.append(targets)
        phis = torch.cat(phis_list)
        targets = torch.cat(targets_list)
        num_classes = logits.size(-1)

        # Get the laplace layer
        for m in self.model.modules():
            if isinstance(m, LaplaceLayer):
                laplace_layer = m

        mean = laplace_layer.layer.weight.data.clone()
        cov = laplace_layer.covariance_matrix.data.clone()
        targets_onehot = F.one_hot(targets, num_classes=num_classes)

        if likelihood == 'gaussian':
            for _ in range(lmb):
                for phi, target_onehot in zip(phis, targets_onehot):
                    # Compute new cov with woodbury identity
                    tmp_1 = torch.matmul(cov, phi)
                    tmp_2 = torch.outer(tmp_1, tmp_1)
                    var = torch.matmul(phi, tmp_1)

                    cov_update = tmp_2 / (1 + var)
                    cov -= cov_update

                    # Update mean
                    logits = F.linear(phi, mean)
                    probas = logits.softmax(-1)

                    tmp_3 = F.linear(gamma*cov, phi)
                    tmp_4 = (target_onehot - probas)
                    mean += torch.outer(tmp_4, tmp_3)

        elif likelihood == 'categorical':
            for _ in range(lmb):
                for phi, target_onehot in zip(phis, targets_onehot):
                    tmp_1 = cov @ phi
                    tmp_2 = torch.outer(tmp_1, tmp_1)

                    # Compute new prediction.
                    var = F.linear(phi, tmp_1)
                    logits = F.linear(phi, mean)
                    probas = logits.softmax(-1)
                    probas_max = probas.max()

                    # Update covariance matrix.
                    num = probas_max * (1-probas_max)
                    denom = 1 + num * var
                    factor = num / denom
                    cov_update = factor * tmp_2
                    cov -= cov_update

                    # Update mean.
                    tmp_3 = F.linear(gamma*cov, phi)
                    tmp_4 = (target_onehot - probas)
                    mean += torch.outer(tmp_4, tmp_3)

                    # Undo cov update.
                    cov += cov_update

                    # Compute new prediction.
                    logits = F.linear(phi, mean)
                    probas = logits.softmax(-1)
                    probas_max = probas.max()

                    # Update covariance matrix.
                    num = probas_max * (1 - probas_max)
                    denom = 1 + num * var
                    factor = num / denom
                    cov_update = factor * tmp_2
                    cov -= cov_update
        else:
            raise NotImplementedError()

        laplace_layer.layer.weight.data = mean
        laplace_layer.covariance_matrix.data = cov
