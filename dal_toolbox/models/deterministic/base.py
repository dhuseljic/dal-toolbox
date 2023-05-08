import lightning as L
import torch.nn.functional as F


class DeterministicModule(L.LightningModule):

    def training_step(self, batch):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        self.log('loss', loss, prog_bar=True)
        return loss
