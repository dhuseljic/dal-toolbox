import warnings

import torch
import torch.nn as nn
import lightning as L


class EnsembleModel(L.LightningModule):
    def __init__(self, members: list, metrics=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.metrics = nn.ModuleDict(metrics)

        self.loss_fn = nn.CrossEntropyLoss()

        self.automatic_optimization = False

    def forward(self, x, **kwargs):
        logits_list = []
        for member in self.members:
            logits_list.append(member(x))
        return torch.stack(logits_list, dim=1)

    def training_step(self, batch):
        inputs, targets = batch
        optimizers = self.optimizers()
        for i_member, (member, optimizer) in enumerate(zip(self.members, optimizers)):
            logits = member(inputs)
            loss = self.loss_fn(logits, targets)

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            self.log(f'loss_member{i_member}', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizers = []
        for member in self.members:
            optimizer = torch.optim.SGD(member.parameters(), lr=1e-1, momentum=.9)
            optimizers.append(optimizer)
        warnings.warn(f'Using default optimizer: {optimizer}.')
        return optimizers
