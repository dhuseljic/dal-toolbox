import copy

import torch
import torch.nn as nn

from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from ..utils.base import BaseModule
from ...metrics import GibbsCrossEntropy, ensemble_log_softmax


class EnsembleModel(BaseModule):
    def __init__(
            self,
            model_list: list,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer_list: list = None,
            lr_scheduler_list: list = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            val_loss_fn=GibbsCrossEntropy(),
    ):
        super(BaseModule, self).__init__()
        self.model = nn.ModuleList(model_list)
        self.loss_fn = loss_fn
        self.optimizer_list = optimizer_list
        self.lr_scheduler_list = lr_scheduler_list

        train_metrics = self._train_metric_per_member(train_metrics)
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)
        self.val_loss_fn = val_loss_fn

        # Save initial stats
        self.init_model_state = copy.deepcopy(self.state_dict())
        if optimizer_list is not None:
            self.init_optimizer_state_list = [copy.deepcopy(opt.state_dict()) for opt in self.optimizer_list]
        if lr_scheduler_list is not None:
            self.init_scheduler_state_list = [copy.deepcopy(lrs.state_dict()) for lrs in self.lr_scheduler_list]

        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        raise ValueError('Ensemble forward is not defined, use `mc_forward`.')

    def mc_forward(self, *args, **kwargs):
        """Returns the logits from each ensemble member.

        Returns:
            torch.Tensor: Tensor of shape (ensemble_size, num_samples, num_classes).
        """
        logits_list = []
        for member in self.model:
            logits_list.append(member(*args, **kwargs))
        return torch.stack(logits_list, dim=1)

    def reset_states(self, reset_model_parameters=True):
        if reset_model_parameters:
            self.load_state_dict(self.init_model_state)
        for optimizer, init_state in zip(self.optimizer_list, self.init_optimizer_state_list):
            optimizer.load_state_dict(init_state)
        if self.lr_scheduler_list:
            for lrs, init_state in zip(self.lr_scheduler_list, self.init_scheduler_state_list):
                lrs.load_state_dict(init_state)
        self.configure_optimizers()

    def _train_metric_per_member(self, train_metrics):
        if train_metrics:
            all_train_metrics = {}
            for i in range(len(self.model)):
                all_train_metrics.update({f'{key}_member{i}': val for key, val in train_metrics.items()})
            return all_train_metrics

    def training_step(self, batch):
        inputs, targets = batch

        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()

        for i, (member, optimizer, lr_scheduler) in enumerate(zip(self.model, optimizers, lr_schedulers)):
            logits = member(inputs)
            loss = self.loss_fn(logits, targets)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            self.log(f'train_loss_member{i}', loss, prog_bar=True)
            if self.trainer.is_last_batch:
                lr_scheduler.step()

            # TODO: How to handle metrics

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        mc_logits = self.mc_forward(inputs)
        loss = self.val_loss_fn(mc_logits, targets)
        self.log('val_loss', loss, prog_bar=True)

        logits = ensemble_log_softmax(mc_logits)
        self.log_val_metrics(logits, targets)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch[0]
        targets = batch[1]
        logits = self.mc_forward(inputs)

        logits = self._gather(logits)
        targets = self._gather(targets)
        return logits, targets

    def configure_optimizers(self):
        if self.optimizer_list is None:
            optimizers = []
            for member in self.model:
                optimizer = torch.optim.SGD(member.parameters(), lr=1e-1, momentum=.9)
                optimizers.append(optimizer)
            rank_zero_warn(f'Using default optimizer for all ensemble members: {optimizer}.')
            return optimizers

        if self.lr_scheduler_list is None:
            return self.optimizer_list

        return self.optimizer_list, self.lr_scheduler_list

    @torch.inference_mode()
    def get_logits(self, *args, **kwargs):
        kwargs['device'] = self.device
        logits_list = []
        for member in self.model:
            if not hasattr(member, 'get_logits'):
                raise NotImplementedError('The `get_logits` method is not implemented.')
            logits_list.append(member.get_logits(*args, **kwargs))
        logits = torch.stack(logits_list, dim=1)
        return logits

    @torch.inference_mode()
    def get_representations(self, *args, **kwargs):
        kwargs['device'] = self.device
        representations_list = []
        for member in self.model:
            if not hasattr(member, 'get_representations'):
                raise NotImplementedError('The `get_representations` method is not implemented.')
            representations_list.append(member.get_representations(*args, **kwargs))
        representations = torch.stack(representations_list, dim=1)
        return representations

    @torch.inference_mode()
    def get_grad_representations(self, *args, **kwargs):
        kwargs['device'] = self.device
        grad_representations_list = []
        for member in self.model:
            if not hasattr(member, 'get_grad_representations'):
                raise NotImplementedError('The `get_grad_representations` method is not implemented.')
            grad_representations_list.append(member.get_grad_representations(*args, **kwargs))
        grad_representations = torch.stack(grad_representations_list, dim=1)
        return grad_representations
