import os
import abc
import copy
import time
import logging
import datetime

import torch
import torch.distributed as dist
import lightning as L

from lightning.pytorch.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter
from ...utils import write_scalar_dict, setup_for_distributed


class BasicTrainer(abc.ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None,
        num_epochs: int = 200,
        device: str = None,
        num_devices: int = 'auto',
        precision='32-true',
        output_dir: str = None,
        summary_writer=None,
        val_every: int = 1,
        save_every: int = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.num_devices = num_devices
        self.precision = precision
        self.output_dir = output_dir
        self.summary_writer = summary_writer
        self.val_every = val_every
        self.save_every = save_every

        self.logger = logging.getLogger(__name__)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(output_dir) if summary_writer is None else summary_writer

        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.init_criterion_state = copy.deepcopy(self.criterion.state_dict())
        if lr_scheduler:
            self.init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())

        self.fabric = L.Fabric(
            accelerator='auto',
            strategy='auto',
            devices=self.num_devices,
            precision=precision,
        )
        self.fabric.launch()
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        setup_for_distributed(self.fabric.global_rank == 0)

        self.train_history: list = []
        self.val_history: list = []
        self.test_stats: dict = {}
        self.cur_epoch = 0

    def reset_states(self, reset_model_parameters=True):
        self.optimizer.load_state_dict(self.init_optimizer_state)
        self.criterion.load_state_dict(self.init_criterion_state)
        if reset_model_parameters:
            self.model.load_state_dict(self.init_model_state)
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.init_scheduler_state)

    @rank_zero_only
    def save_checkpoint(self, i_epoch=None, fname="checkpoint.pth"):
        self.logger.info('Saving %s..', fname)
        start_time = time.time()
        checkpoint_path = os.path.join(self.output_dir, fname)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": i_epoch,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            # "train_history": self.train_history,
            # "test_history": self.test_history,
        }
        self.fabric.save(checkpoint_path, checkpoint)
        saving_time = (time.time() - start_time)
        self.logger.info('Saving took %s', str(datetime.timedelta(seconds=int(saving_time))))

    def fit(self, train_loader, val_loaders=None):
        self.logger.info('Training with %s instances..', len(train_loader.dataset))
        start_time = time.time()

        train_loader = self.fabric.setup_dataloaders(train_loader)

        self.train_history = []
        self.val_history = []
        self.model.to(self.device)
        for i_epoch in range(1, self.num_epochs+1):
            self.cur_epoch = i_epoch

            train_stats = self.train_one_epoch(dataloader=train_loader, epoch=i_epoch)

            # Logging
            self.train_history.append(train_stats)
            if self.summary_writer is not None:
                write_scalar_dict(self.summary_writer, train_stats, prefix='train', global_step=i_epoch)

            # Validate in intervals if val loader exists
            if val_loaders and i_epoch % self.val_every == 0:
                # TODO(dhuseljic): validation not model agnostic
                if isinstance(val_loaders, dict):
                    val_loader = val_loaders.get('val_loader')
                    val_loader_ood = val_loaders.get('val_loaders_ood')
                else:
                    val_loader = val_loaders
                    val_loader_ood = None
                val_stats = self.evaluate(dataloader=val_loader, dataloaders_ood=val_loader_ood)
                self.val_history.append(val_stats)

            # Save checkpoint in intervals if output directory is defined
            if self.output_dir and self.save_every and i_epoch % self.save_every == 0:
                self.save_checkpoint(i_epoch)

        training_time = (time.time() - start_time)
        self.logger.info('Training took %s', str(datetime.timedelta(seconds=int(training_time))))
        self.logger.info('Training stats of final epoch: %s', train_stats)

        # Save final model if output directory is defined
        if self.output_dir is not None:
            self.logger.info('Saving final model..')
            self.save_checkpoint(i_epoch, fname='model_final.pth')

        return {'train_history': self.train_history, 'test_history': self.val_history}

    @torch.no_grad()
    def evaluate(self, dataloader, dataloaders_ood: dict = None):
        self.logger.info('Evaluation with %s instances..', len(dataloader.dataset))
        if dataloaders_ood:
            for name, dl in dataloaders_ood.items():
                self.logger.info('> OOD dataset %s with %s instances..', name, len(dl.dataset))
        start_time = time.time()
        test_stats = self.evaluate_model(dataloader, dataloaders_ood)
        eval_time = (time.time() - start_time)

        if self.summary_writer is not None:
            write_scalar_dict(self.summary_writer, test_stats, prefix='test', global_step=self.cur_epoch)

        self.logger.info('Evaluation stats: %s', test_stats)
        self.logger.info('Evaluation took %s', str(datetime.timedelta(seconds=int(eval_time))))
        return test_stats

    def backward(self, loss):
        self.fabric.backward(loss)
        # loss.backward()

    def step_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def all_gather(self, val):
        if not dist.is_available() or not dist.is_initialized():
            return val
        gathered_vals = self.fabric.all_gather(val)
        val = torch.cat([v for v in gathered_vals])
        # Pure pytorch gather:
        # gathered_vals = [torch.zeros_like(val) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_vals, val)
        # val = torch.cat(gathered_vals)
        return val

    @abc.abstractmethod
    def train_one_epoch(self, dataloader, epoch):
        pass

    @abc.abstractmethod
    def evaluate_model(self, dataloader, dataloaders_ood):
        pass

    def predict(self, dataloader):
        raise NotImplementedError('Predict method is not implemented.')
