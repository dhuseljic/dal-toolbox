from .base import DeterministicModel
from . import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class SimCLR(DeterministicModel):
    def __init__(
        self,
        hidden_dim: int,
        temperature: float,
        optimizer: torch.optim.Optimizer = None,
        optimizer_params: dict = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        lr_scheduler_params: dict = None,
        train_metrics: dict = None,
        val_metrics: dict = None,
    ):
        model = resnet.ResNet18(4*hidden_dim) # num_classes is the output size of the last linear layer
        super().__init__(model, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, train_metrics, val_metrics)

        assert temperature > 0.0, "The temperature must be a positive float!"
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.model.linear = nn.Sequential(
            self.model.linear,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.temperature = temperature


    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.model(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "/loss", nll, sync_dist=True)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "/acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        self.log(mode + "/acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        self.log(mode + "/acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_train_epoch_end(self) -> None:
        logging.info("Current Performance-Metric-Values:")
        for metr, val in self.trainer.logged_metrics.items():
            logging.info(metr+" : "+str(round(val.item(), 5)))
        return super().on_train_epoch_end()