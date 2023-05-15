from ..utils.base import BaseModule


class MCDropoutModel(BaseModule):

    def mc_forward(self, *args, **kwargs):
        return self.model.mc_forward(*args, **kwargs)

    def training_step(self, batch):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss, prog_bar=True)

        if self.train_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.train_metrics.items()}
            self.log_dict(self.train_metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        if self.val_metrics is not None:
            metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.val_metrics.items()}
            self.log_dict(self.val_metrics, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch[0]
        targets = batch[1]
        logits = self.model.mc_forward(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets
