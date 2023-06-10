from .base import DeterministicModel
from . import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


# TODO This should probably be somewhere else
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()

        assert temperature > 0.0, "The temperature must be a positive float!"
        self.temperature = temperature

    def forward(self, batch, targets):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(batch[:, None, :], batch[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        infoNCE = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        infoNCE = infoNCE.mean()

        return infoNCE


class SimCLR(DeterministicModel):
    def __init__(
            self,
            encoder,
            projector,
            log_on_epoch_end=True,
            loss_fn: nn.Module = InfoNCELoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        model = nn.Sequential(encoder, projector)

        super().__init__(model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         train_metrics=train_metrics, val_metrics=val_metrics)

        self.encoder = encoder
        self.projector = projector
        self.log_on_epoch_end = log_on_epoch_end

    def training_step(self, batch):
        batch[0] = torch.cat(batch[0], dim=0)
        return super().training_step(batch)

    def validation_step(self, batch, batch_idx):
        batch[0] = torch.cat(batch[0], dim=0)
        super().validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        if self.log_on_epoch_end:
            log_str = "Current Performance-Metric-Values: "
            for metr, val in self.trainer.logged_metrics.items():
                log_str += (metr + " : " + str(round(val.item(), 5)) + ", ")
            logging.info(log_str)
        return super().on_train_epoch_end()
