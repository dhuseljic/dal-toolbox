import inspect

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.laplace import LaplaceLinear
from .base import BaseModule


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
    ):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, train_metrics, val_metrics, scheduler_interval)
        # self.use_mean_field = True
        self.predict_type = 'mean_field'

    def training_step(self, batch):
        inputs = batch[0]
        targets = batch[1]

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

        if self.predict_type == 'mean_field':
            logits = self.model.forward_mean_field(inputs)
        elif self.predict_type == 'monte_carlo':
            logits = self.model.forward_monte_carlo(inputs)
        elif self.predict_type == 'deterministic':
            logits = self.model(inputs)
        else:
            raise NotImplementedError()
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        self.log_val_metrics(logits, targets)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch

        if self.predict_type == 'mean_field':
            logits = self.model.forward_mean_field(inputs)
        elif self.predict_type == 'monte_carlo':
            logits = self.model.forward_monte_carlo(inputs)
        elif self.predict_type == 'deterministic':
            logits = self.model(inputs)
        else:
            raise NotImplementedError()

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def set_mc_samples(self, mc_samples: int):
        for m in self.model.modules():
            if isinstance(m, LaplaceLinear):
                m.mc_samples = mc_samples

    def set_mean_field_factor(self, mean_field_factor: float):
        for m in self.model.modules():
            if isinstance(m, LaplaceLinear):
                m.mean_field_factor = mean_field_factor

    def _reset_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, LaplaceLinear):
                m.reset_precision_matrix()

    def _synchronize_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, LaplaceLinear):
                m.synchronize_precision_matrix()

    @torch.no_grad()
    def get_variances(self, *args, **kwargs):
        if not hasattr(self.model, 'get_variances'):
            raise NotImplementedError('The `get_variances` method is not implemented.')
        return self.model.get_variances(*args, **kwargs)

    @torch.no_grad()
    def update_posterior(
        self,
        dataloader,
        lmb=1,
        gamma=10,
        cov_likelihood='gaussian',
        update_type='second_order',
        from_features=False,
        optimizer=None,
        device='cpu'
    ):
        # Set to eval to avoid precision matrix update
        self.eval()
        self.to(device)

        # check if return_features is in forward_kwargs
        if not hasattr(self.model, 'forward_features') or not hasattr(self.model, 'forward_head'):
            raise ValueError('The methods `forward_features` and `forward_head` need to be defined.')

        phis_list = []
        targets_list = []
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.unsqueeze(0) if inputs.ndim == 1 else inputs
            targets = targets.unsqueeze(0) if targets.ndim == 0 else targets

            if from_features:
                phis = inputs
                logits = self.model.forward_head(phis, mean_field=True)
            else:
                phis = self.model.forward_features(inputs)
                logits = self.model.forward_head(phis, mean_field=True)
            phis_list.append(phis)
            targets_list.append(targets)
        phis = torch.cat(phis_list)
        targets = torch.cat(targets_list)
        num_classes = logits.size(-1)

        # Get the laplace layer
        for m in self.model.modules():
            if isinstance(m, LaplaceLinear):
                laplace_layer = m

        laplace_layer.compute_covariance()
        mean = laplace_layer.layer.weight.data
        cov = laplace_layer.covariance_matrix.data
        targets_onehot = F.one_hot(targets, num_classes=num_classes)

        if update_type == 'first_order':
            new_mean = first_order_update(phis, targets_onehot, mean, gamma=gamma, lmb=lmb)
            new_cov = cov.clone()
        elif update_type == 'second_order':
            new_mean, new_cov = second_order_update(
                phis, targets_onehot, mean, cov, cov_likelihood=cov_likelihood, gamma=gamma, lmb=lmb)
        elif update_type == 'optimizer':
            if optimizer is None:
                raise ValueError('You need to set the `optimizer` argument to use it for updating.')
            with torch.enable_grad():
                linear = copy.deepcopy(laplace_layer.layer)
                for phi, target in zip(phis, targets):
                    loss = F.cross_entropy(linear(phi), target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                new_mean = linear.weight.data
            new_cov = cov.clone()
        laplace_layer.layer.weight.data = new_mean
        laplace_layer.covariance_matrix.data = new_cov


def first_order_update(features, targets_onehot, mean, gamma=1, lmb=1):
    mean = mean.clone()
    for _ in range(lmb):
        for feature, target_onehot in zip(features, targets_onehot):
            logits = F.linear(feature, mean)
            probas = logits.softmax(-1)

            tmp_3 = gamma*feature.clone()
            tmp_4 = (target_onehot - probas)
            mean += torch.outer(tmp_4, tmp_3)
    return mean


def kfac_update(features, targets_onehot, mean, gamma=1, lmb=1):
    mean = mean.clone()

    alpha = 0.9
    A_running = torch.zeros(features.size(-1), features.size(-1))
    G_running = torch.zeros(mean.size(0), mean.size(0))

    for _ in range(lmb):
        for feature, target_onehot in zip(features, targets_onehot):

            logits = F.linear(feature, mean)
            probas = logits.softmax(-1)

            grad = probas - target_onehot
            A = torch.outer(feature, feature)
            G = torch.outer(grad, grad)

            A_running = alpha * A_running + (1-alpha) * A
            G_running = alpha * G_running + (1-alpha) * G

            damping = 1e-3
            A = A_running + damping * torch.eye(feature.size(-1))
            G = G_running + damping * torch.eye(probas.size(-1))

            A_inv = torch.inverse(A)
            G_inv = torch.inverse(G)

            grad_mean = torch.outer(grad, feature)
            update = G_inv @ grad_mean @ A_inv
            mean -= gamma*update
    return mean


def second_order_update(features, targets_onehot, mean, cov, cov_likelihood='gaussian', gamma=1, lmb=1):
    mean = mean.clone()
    cov = cov.clone()

    if cov_likelihood == 'gaussian':
        for _ in range(lmb):
            for feature, target_onehot in zip(features, targets_onehot):
                # Compute new cov with woodbury identity
                tmp_1 = torch.matmul(cov, feature)
                tmp_2 = torch.outer(tmp_1, tmp_1)
                var = torch.matmul(feature, tmp_1)

                cov_update = tmp_2 / (1 + var)
                cov -= cov_update

                # Update mean
                logits = F.linear(feature, mean)
                probas = logits.softmax(-1)

                tmp_3 = F.linear(gamma*cov, feature)
                tmp_4 = (target_onehot - probas)
                mean += torch.outer(tmp_4, tmp_3)
    elif cov_likelihood == 'categorical':
        for _ in range(lmb):
            for feature, target_onehot in zip(features, targets_onehot):
                tmp_1 = cov @ feature
                tmp_2 = torch.outer(tmp_1, tmp_1)

                # Compute new prediction.
                var = F.linear(feature, tmp_1)
                logits = F.linear(feature, mean)
                probas = logits.softmax(-1)
                probas_max = probas.max()

                # Update covariance matrix.
                num = probas_max * (1-probas_max)
                # denom = 1 + num * var
                denom = 1 + var
                factor = num / denom
                cov_update = factor * tmp_2
                cov -= cov_update

                # Update mean.
                tmp_3 = F.linear(gamma*cov, feature)
                tmp_4 = (target_onehot - probas)
                mean += torch.outer(tmp_4, tmp_3)

                # Undo cov update.
                cov += cov_update

                # Compute new prediction.
                logits = F.linear(feature, mean)
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

    return mean, cov
