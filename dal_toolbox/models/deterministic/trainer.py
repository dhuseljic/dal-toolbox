import os
import copy
import time
import logging

import torch

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

logger = logging.getLogger(__name__)


class BasicTrainer:

    def __init__(self, model, optimizer, criterion, train_one_epoch, evaluate, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.train_one_epoch = train_one_epoch
        self.evaluate_fn = evaluate

        self.device = device
        self.use_distributed = use_distributed

        self.summary_writer = summary_writer
        self.output_dir = output_dir

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
        self.model.to(self.device)

        t1 = time.time()

        self.train_history = []
        self.test_history = []
        for i_epoch in range(1, n_epochs+1):
            if self.use_distributed:
                train_loader.sampler.set_epoch(i_epoch)
            train_stats = self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                epoch=i_epoch,
                device=self.device
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.train_history.append(train_stats)

            # Logging
            if self.summary_writer is not None:
                self.write_scalar_dict(train_stats, prefix='train', global_step=i_epoch)

            # Eval in intervals if test loader exists
            if test_loaders and i_epoch % eval_every == 0:
                logger.info('Evaluation ...')
                test_loader_id = test_loaders.get('test_loader_id')
                test_loader_ood = test_loaders.get('test_loader_ood', {})
                t1 = time.time()
                test_stats = self.evaluate(test_loader_id=test_loader_id, test_loader_ood=test_loader_ood)
                logger.info(test_stats)
                self.test_history.append(test_stats)
                logger.info('Evaluation took %.2f minutes', (time.time() - t1)/60)

            # Save checkpoint in intervals if output directory is defined
            if self.output_dir and save_every and i_epoch % save_every == 0:
                t1 = time.time()
                logger.info('Saving checkpoint')
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": i_epoch,
                    "train_history": self.train_history,
                    "test_history": self.test_history,
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                }
                torch.save(checkpoint, os.path.join(self.output_dir, "checkpoint.pth"))
                logger.info('Saving took %.2f minutes', (time.time() - t1)/60)

        training_time = (time.time() - t1)
        logger.info('Training took %.2f minutes', training_time/60)
        logger.info('Training stats: %s', train_stats)

        # Save final model if output directory is defined
        if self.output_dir is not None:
            t1 = time.time()
            fname = os.path.join(self.output_dir, "model_final.pth")
            logger.info('Saving final model to %s', fname)
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": i_epoch,
                "train_history": self.train_history,
                "test_history": self.test_history,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }
            torch.save(checkpoint, fname)
            logger.info('Saving took %.2f minutes', (time.time() - t1)/60)

        return {'train_history': self.train_history, 'test_history': self.test_history}

    def evaluate(self, test_loader_id, test_loader_ood={}):
        self.test_stats = self.evaluate_fn(
            model=self.model,
            dataloader_id=test_loader_id,
            dataloaders_ood=test_loader_ood,
            criterion=self.criterion,
            device=self.device
        )
        return self.test_stats

    def write_scalar_dict(self, scalar_dict, prefix, global_step):
        if self.summary_writer is not None:
            for key, val in scalar_dict.items():
                self.summary_writer.add_scalar(f'{prefix}/{key}', val, global_step=global_step)
