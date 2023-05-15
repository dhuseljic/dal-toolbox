import abc

import torch
import torch.nn as nn
import torch.distributed as dist

import lightning as L

from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class BaseModule(L.LightningModule, abc.ABC):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer = None,
            # optimizer_params: dict = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            # lr_scheduler_params: dict = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            loss_fn=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        # self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        # self.lr_scheduler_params = lr_scheduler_params
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)
        self.loss_fn = loss_fn

        # TODO(dhuseljic): optimizer and lrscheduler as args?
        # init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        # init_model_state = copy.deepcopy(self.model.state_dict())
        # init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _gather(self, val):
        if not dist.is_available() or not dist.is_initialized():
            return val
        gathered_val = self.all_gather(val)
        val = torch.cat([v for v in gathered_val])
        return val

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-1, momentum=.9, weight_decay=0.01)
            rank_zero_warn(f'Using default optimizer: {optimizer}.')
            return optimizer
        if self.lr_scheduler is None:
            return self.optimizer
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
        # else:
        #     optimizer_params = {} if self.optimizer_params is None else self.optimizer_params
        #     optimizer = self.optimizer(self.parameters(), **optimizer_params)
        # if self.lr_scheduler is None:
        #     return optimizer
        # lr_scheduler_params = {} if self.lr_scheduler_params is None else self.lr_scheduler_params
        # lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_params)
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    # TODO(dhuseljic): Discuss
    @torch.inference_mode()
    def get_logits(self, *args, **kwargs):
        kwargs['device'] = self.device
        if not hasattr(self.model, 'get_logits'):
            raise NotImplementedError('The `get_logits` method is not implemented.')
        return self.model.get_logits(*args, **kwargs)

    @torch.inference_mode()
    def get_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_representations'):
            raise NotImplementedError('The `get_representations` method is not implemented.')
        return self.model.get_representations(*args, **kwargs)

    @torch.inference_mode()
    def get_grad_representations(self, *args, **kwargs):
        if not hasattr(self.model, 'get_grad_representations'):
            raise NotImplementedError('The `get_grad_representations` method is not implemented.')
        return self.model.get_grad_representations(*args, **kwargs)
