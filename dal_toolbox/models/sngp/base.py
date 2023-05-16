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
