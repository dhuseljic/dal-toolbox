import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning as L

from ..utils.random_features import RandomFeatureGaussianProcess


class SNGPModel(L.LightningModule):
    def __init__(self, model, metrics=None):
        super().__init__()
        self.model = model
        self.metrics = nn.ModuleDict(metrics)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True)

        if self.metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.metrics.items()}
            self.log_dict(self.metrics, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.reset_precision_matrix()

    def on_train_epoch_end(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.synchronize_precision_matrix()

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
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=.9, weight_decay=0.01)
        warnings.warn(f'Using default optimizer: {optimizer}.')
        return optimizer

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
