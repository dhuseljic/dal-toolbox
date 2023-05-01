import os
import abc
import copy
import time
import logging
import datetime

import torch

import lightning as L

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from ...utils import write_scalar_dict


class BasicTrainer(abc.ABC):
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 lr_scheduler=None,
                 device=None,
                 output_dir=None,
                 summary_writer=None,
                 num_gpus='auto',
                 use_distributed=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.device = device
        self.use_distributed = use_distributed

        self.logger = logging.getLogger(__name__)
        self.summary_writer = summary_writer
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if self.use_distributed:
            self.logger("Using distributed mode.")
            self.model.to(device)
            rank = torch.distributed.get_rank()
            self.model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)

        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.init_criterion_state = copy.deepcopy(self.criterion.state_dict())
        if lr_scheduler:
            self.init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())


        self.num_gpus = torch.cuda.device_count() if num_gpus == 'auto' else num_gpus
        self.fabric = L.Fabric(accelerator='cuda', devices=self.num_gpus, strategy='ddp')
        self.fabric.launch()

        self.train_history: list = []
        self.test_history: list = []
        self.test_stats: dict = {}

    def reset_states(self, reset_model_parameters=False):
        self.optimizer.load_state_dict(self.init_optimizer_state)
        self.criterion.load_state_dict(self.init_criterion_state)
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.init_scheduler_state)
        if reset_model_parameters:
            self.model.load_state_dict(self.init_model_state)

    def save_checkpoint(self, i_epoch=None):
        self.logger.info('Saving checkpoint..')
        start_time = time.time()
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth")
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": i_epoch,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            # "train_history": self.train_history,
            # "test_history": self.test_history,
        }
        torch.save(checkpoint, checkpoint_path)
        saving_time = (time.time() - start_time)
        self.logger.info('Saving took %s', str(datetime.timedelta(seconds=int(saving_time))))


    def train(self, n_epochs, train_loader, test_loaders=None, eval_every=None, save_every=None):
        self.logger.info('Training with %s instances..', len(train_loader.dataset))
        start_time = time.time()

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)

        if self.use_distributed:
            if not isinstance(train_loader.sampler, DistributedSampler):
                raise ValueError('Configure a distributed sampler to use distributed training.')

        self.train_history = []
        self.test_history = []
        self.model.to(self.device)
        for i_epoch in range(1, n_epochs+1):
            if self.use_distributed:
                train_loader.sampler.set_epoch(i_epoch)

            train_stats = self.train_one_epoch(dataloader=train_loader, epoch=i_epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.train_history.append(train_stats)

            # Logging
            if self.summary_writer is not None:
                write_scalar_dict(train_stats, prefix='train', global_step=i_epoch)

            # Eval in intervals if test loader exists
            if test_loaders and i_epoch % eval_every == 0:
                test_loader = test_loaders.get('test_loader')
                test_loaders_ood = test_loaders.get('test_loaders_ood')
                test_stats = self.evaluate(dataloader=test_loader, dataloaders_ood=test_loaders_ood)
                self.test_history.append(test_stats)

            # Save checkpoint in intervals if output directory is defined
            if self.output_dir and save_every and i_epoch % save_every == 0:
                self.save_checkpoint(i_epoch)

        training_time = (time.time() - start_time)
        self.logger.info('Training took %s', str(datetime.timedelta(seconds=int(training_time))))
        self.logger.info('Training stats of final epoch: %s', train_stats)

        # Save final model if output directory is defined
        if self.output_dir is not None:
            self.save_checkpoint(i_epoch)

        return {'train_history': self.train_history, 'test_history': self.test_history}

    @torch.no_grad()
    def evaluate(self, dataloader, dataloaders_ood: dict = None):
        self.logger.info('Evaluation with %s instances..', len(dataloader.dataset))
        if dataloaders_ood:
            for name, dl in dataloaders_ood.items():
                self.logger.info('> OOD dataset %s with %s instances..', name, len(dl.dataset))
        start_time = time.time()
        test_stats = self.evaluate_model(dataloader, dataloaders_ood)
        eval_time = (time.time() - start_time)
        self.logger.info('Evaluation stats: %s', test_stats)
        self.logger.info('Evaluation took %s', str(datetime.timedelta(seconds=int(eval_time))))
        return test_stats

    @torch.no_grad()
    def collect_predictions(self, dataloader):
        all_logits = []
        all_targets = []
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            all_logits.append(self.model(inputs).cpu())
            all_targets.append(targets)
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return logits, targets

    @abc.abstractmethod
    def train_one_epoch(self, dataloader, epoch):
        pass

    @abc.abstractmethod
    def evaluate_model(self, dataloader, dataloaders_ood):
        pass
