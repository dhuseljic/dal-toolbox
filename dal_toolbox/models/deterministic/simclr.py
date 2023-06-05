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
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # TODO Figure out where logging goes
        # Logging loss
        # self.log(mode + "/loss", nll, sync_dist=True)
        # Get ranking position of positive example
        # comb_sim = torch.cat(
        #     [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        #    dim=-1,
        # )
        # sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        # self.log(mode + "/acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        # self.log(mode + "/acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        # self.log(mode + "/acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return nll


class SimCLR(DeterministicModel):
    def __init__(
            self,
            model,
            loss_fn: nn.Module = InfoNCELoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            hidden_dim: int = 128,
            train_metrics: dict = None,
            val_metrics: dict = None,
    ):
        super().__init__(model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         train_metrics=train_metrics, val_metrics=val_metrics)

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.model.linear = nn.Sequential(
            self.model.linear,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def training_step(self, batch):
        batch[0] = torch.cat(batch[0], dim=0)
        return super().training_step(batch)

    def validation_step(self, batch, batch_idx):
        batch[0] = torch.cat(batch[0], dim=0)
        super().validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        logging.info("Current Performance-Metric-Values:")
        for metr, val in self.trainer.logged_metrics.items():
            logging.info(metr + " : " + str(round(val.item(), 5)))
        return super().on_train_epoch_end()
