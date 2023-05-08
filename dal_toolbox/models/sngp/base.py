import torch
import torch.nn.functional as F
import lightning as L


class SNGPModule(L.LightningModule):

    def on_train_epoch_start(self):
        self.last.reset_precision_matrix()

    def training_step(self, batch):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        self.log('loss', loss)
        return loss

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
