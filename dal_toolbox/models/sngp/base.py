import inspect

import torch
import torch.nn.functional as F

from ..utils.random_features import RandomFeatureGaussianProcess
from ..utils.base import BaseModule


class SNGPModel(BaseModule):

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

    def validation_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        self.log_val_metrics(logits, targets)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO(dhuseljic): discuss with mherde; maybe can be used for AL?
        inputs, targets = batch
        logits = self(inputs, mean_field=True)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def _reset_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.reset_precision_matrix()

    def _synchronize_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.synchronize_precision_matrix()

    def _log_marg_likelihood(self, inputs, targets, prior_precision=2000):
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

    def update_posterior(self, dataloader, lmb=1):
        # TODO(dhuseljic): what to do, when scale features is False, is the updating still correct?
        self.eval()

        # check if return_random_features is in forward_kwargs
        forward_kwargs = inspect.signature(self.model.forward).parameters
        if 'return_random_features' not in forward_kwargs:
            raise ValueError('Define the kwarg `return_random_features` in the forward method of your model.')

        # get rff features
        phis_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                logits, phis = self.model(inputs, return_random_features=True)
                phis_list.append(phis)
                targets_list.append(targets)
        phis = torch.cat(phis_list)
        targets = torch.cat(targets_list)
        num_classes = logits.size(-1)

        # Get the random feature layer
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                random_feature_gp = m

        mean = random_feature_gp.beta.weight.data.clone()
        cov = random_feature_gp.covariance_matrix.data.clone()
        targets_onehot = F.one_hot(targets, num_classes=num_classes)

        for phi, target_onehot in zip(phis, targets_onehot):
            for _ in range(lmb):
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
                tmp_3 = F.linear(cov, phi)
                tmp_4 = (target_onehot - probas)
                mean += torch.outer(tmp_4, tmp_3)

                # Undo cov update.
                cov += cov_update

                # Compute new prediction.
                var = F.linear(phi, tmp_1)
                logits = F.linear(phi, mean)
                probas = logits.softmax(-1)
                probas_max = probas.max()

                # Update covariance matrix.
                num = probas_max * (1 - probas_max)
                denom = 1 + num * var
                factor = num / denom
                cov_update = factor * tmp_2
                cov -= cov_update

        random_feature_gp.beta.weight.data = mean
        random_feature_gp.covariance_matrix.data = cov
        # model_reweighted = copy.deepcopy(model)
        # model_reweighted.model.last.beta.weight.data = mean
        # model_reweighted.model.last.covariance_matrix.data = cov
