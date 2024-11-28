import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L



class SimCLR(L.LightningModule):
    def __init__(self, 
                 encoder, 
                 projector, 
                 temperature = 1.0,
                 model_dir='./pretrained_models/', 
                 random_seed=42, 
                 n_epochs=200, 
                 optimizer_args = {'lr':1e-2, 'weight_decay':5e-4, 'nesterov':True, 'momentum':0.9}):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_func = InfoNCELoss(temperature=temperature)
        self.best_val_loss = None
        self.random_seed = random_seed
        self.model_dir = model_dir
        self.n_epochs = n_epochs
        self.optimizer_args = optimizer_args

    def training_step(self, batch, idx):
        samples, _ = batch
        samples = torch.cat(samples, dim=0)

        features = self.encoder(samples)
        projections = self.projector(features)

        loss = self.loss_func(projections)

        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, idx):
        samples, _ = batch
        samples = torch.cat(samples, dim=0)

        features = self.encoder(samples)
        projections = self.projector(features)

        loss = self.loss_func(projections)

        self.log_dict({'val_loss': loss})

        return loss
    
    def on_validation_epoch_end(self):
        # Check for optimal val loss and save model dict accordingly.
        curr_val_loss = self.trainer.callback_metrics['val_loss'].item()
        if not self.best_val_loss or curr_val_loss < self.best_val_loss:
            self.best_val_loss = curr_val_loss
            path = os.path.join(self.model_dir, "pretrained_weights_seed_"+str(self.random_seed)+".pth")
            logging.info(f"Saving new best model with a validation loss of {curr_val_loss} after Epoch {self.current_epoch} to {path}!")
            state_dict = self.encoder.state_dict()
            torch.save(state_dict, path)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
                params=self.parameters(), 
                **self.optimizer_args
                )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.n_epochs)
        return [optimizer], [lr_scheduler]





class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        assert temperature > 0.0, "The temperature must be a positive float!"
        self.temperature = temperature

    def forward(self, batch):
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