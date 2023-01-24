import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        raise ValueError('Forward method should only be used on ensemble members.')

    def forward_sample(self, x):
        logits = []
        for m in self.models:
            logits.append(m(x))
        return torch.stack(logits)

    def __iter__(self):
        for m in self.models:
            yield m

    def __len__(self):
        return len(self.models)


class EnsembleLRScheduler:
    def __init__(self, lr_schedulers: list):
        self.lr_schedulers = lr_schedulers

    def step(self):
        for lrs in self.lr_schedulers:
            lrs.step()

    def state_dict(self) -> dict:
        return [lrs.state_dict() for lrs in self.lr_schedulers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for lrs, state_dict in zip(self.lr_schedulers, state_dict_list):
            lrs.load_state_dict(state_dict)

    def __iter__(self):
        for lrs in self.lr_schedulers:
            yield lrs


class EnsembleOptimizer:
    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    def state_dict(self) -> dict:
        return [optim.state_dict() for optim in self.optimizers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for optim, state_dict in zip(self.optimizers, state_dict_list):
            optim.load_state_dict(state_dict)

    def __iter__(self):
        for optim in self.optimizers:
            yield optim
