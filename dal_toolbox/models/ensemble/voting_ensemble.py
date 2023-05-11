import copy

import torch
import torch.nn as nn

from torch.func import stack_module_state, functional_call


class Ensemble(nn.Module):
    def __init__(self, members: list):
        super().__init__()
        # self.members = nn.ModuleList(models)
        self.members = nn.ModuleList(members)

        self.num_members = len(members)
        # self.vmap_setup = False
        # self.vmap_randomness = 'different'

    # def _setup_vmap(self):
    #     if self.vmap_setup is False:
    #         self.params, self.buffers = stack_module_state(self.members)
    #         base_model = copy.deepcopy(self.members[0])
    #         base_model = base_model.to('meta')
    #         self._f = lambda params, buffers, x: functional_call(base_model, (params, buffers), (x,))
    #         self._forward_ensemble = torch.vmap(self._f, in_dims=(0, 0, None), randomness=self.vmap_randomness)
    #         self.vmap_setup = True

    def forward(self, x):
        raise ValueError('Use forward sample to obtain ensemble predictions.')

    # def forward_sample(self, x):
    #     self._setup_vmap()
    #     logits = self._forward_ensemble(self.params, self.buffers, x)
    #     # logits = self._forward_ensemble(*stack_module_state(self.members), x)
    #     return logits.permute(1, 0, 2)

    def forward_sample(self, x):
        logits = []
        for m in self.members:
            logits.append(m(x))
        logits = torch.stack(logits, dim=1)
        return logits

    def __iter__(self):
        # self.vmap_setup = False
        for m in self.members:
            yield m

    def __len__(self):
        return len(self.members)

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = self.forward_sample(samples.to(device))
            all_logits.append(logits.cpu())
        logits = torch.cat(all_logits, dim=0)
        probas = logits.softmax(-1)
        return probas


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

    def __len__(self):
        return len(self.optimizers)
