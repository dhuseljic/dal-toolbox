from ..utils.base import BaseModule


class MCDropoutModel(BaseModule):

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
        # TODO(dhuseljic): Validation with MC forward? might take a long time
        inputs, targets = batch

        logits = self(inputs)
        loss = self.loss_fn(logits, targets)

        self.log('val_loss', loss, prog_bar=True)
        self.val_metrics(logits, targets)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch[0]
        targets = batch[1]
        logits = self.model.mc_forward(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets
