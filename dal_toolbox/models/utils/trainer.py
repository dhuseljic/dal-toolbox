import os
import abc
import copy
import time
import logging

import torch

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


class BasicTrainer(abc.ABC):
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 lr_scheduler=None,
                 device=None,
                 output_dir=None,
                 summary_writer=None,
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.use_distributed:
            self.model.to(device)
            rank = int(os.environ["LOCAL_RANK"])
            self.model = DistributedDataParallel(model, device_ids=[rank])

        self.init_model_state = copy.deepcopy(self.model.state_dict())
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.init_criterion_state = copy.deepcopy(self.criterion.state_dict())
        self.init_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())
        # TODO: write reset to reset these states

        self.train_history: list = []
        self.test_history: list = []
        self.test_stats: dict = {}

    def train(self, n_epochs, train_loader, test_loaders=None, eval_every=None, save_every=None):
        if self.use_distributed:
            if not isinstance(train_loader.sampler, DistributedSampler):
                raise ValueError('Configure a distributed sampler to use distributed training.')
        self.logger.info('Training with %s instances..', len(train_loader.dataset))
        t1 = time.time()

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
                self.write_scalar_dict(train_stats, prefix='train', global_step=i_epoch)

            # Eval in intervals if test loader exists
            if test_loaders and i_epoch % eval_every == 0:
                test_loader = test_loaders.get('test_loader')
                test_loaders_ood = test_loaders.get('test_loaders_ood')
                test_stats = self.evaluate(dataloader=test_loader, dataloaders_ood=test_loaders_ood)
                self.test_history.append(test_stats)

            # Save checkpoint in intervals if output directory is defined
            if self.output_dir and save_every and i_epoch % save_every == 0:
                self.save_checkpoint(i_epoch)

        training_time = (time.time() - t1)
        self.logger.info('Training took %.2f minutes', training_time/60)
        self.logger.info('Training stats: %s', train_stats)

        # Save final model if output directory is defined
        if self.output_dir is not None:
            self.save_checkpoint(i_epoch)

        return {'train_history': self.train_history, 'test_history': self.test_history}

    def save_checkpoint(self, i_epoch=None):
        self.logger.info('Saving checkpoint..')
        t1 = time.time()
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
        self.logger.info('Saving took %.2f minutes', (time.time() - t1)/60)

    def write_scalar_dict(self, scalar_dict, prefix, global_step):
        if self.summary_writer is not None:
            for key, val in scalar_dict.items():
                self.summary_writer.add_scalar(f'{prefix}/{key}', val, global_step=global_step)

    @abc.abstractmethod
    def train_one_epoch(self, dataloader, epoch):
        pass

    @abc.abstractmethod
    def evaluate(self, dataloader, dataloaders_ood):
        pass
