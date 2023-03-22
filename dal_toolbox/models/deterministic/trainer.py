import os
import copy
import time
import logging

import torch

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from ...metrics import generalization, calibration, ood
from ...utils import MetricLogger, SmoothedValue


logger = logging.getLogger(__name__)


class DeterministicTrainer:

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
                test_stats = self.evaluate(model=self.model, dataloader_id=test_loader_id,
                                           dataloaders_ood=test_loader_ood)
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

    def train_one_epoch(self, model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
        model.train()
        model.to(device)
        criterion.to(device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

        return train_stats

    @torch.no_grad()
    def evaluate(self, model, dataloader_id, dataloaders_ood):
        model.eval()
        model.to(self.device)

        # Forward prop in distribution
        logits_id, targets_id, = [], []
        for inputs, targets in dataloader_id:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits_id.append(model(inputs))
            targets_id.append(targets)
        logits_id = torch.cat(logits_id, dim=0).cpu()
        targets_id = torch.cat(targets_id, dim=0).cpu()

        # Confidence- and entropy-Scores of in domain set logits
        probas_id = logits_id.softmax(-1)
        conf_id, _ = probas_id.max(-1)
        entropy_id = ood.entropy_fn(probas_id)

        # Model specific test loss and accuracy for in domain testset
        acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
        prec = generalization.avg_precision(probas_id, targets_id)
        loss = self.criterion(logits_id, targets_id).item()

        # Negative Log Likelihood
        nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

        # Top- and Marginal Calibration Error
        tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
        mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

        metrics = {
            "acc1": acc1,
            "prec": prec,
            "loss": loss,
            "nll": nll,
            "tce": tce,
            "mce": mce
        }

        for name, dataloader_ood in dataloaders_ood.items():
            # Forward prop out of distribution
            logits_ood = []
            for inputs, targets in dataloader_ood:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits_ood.append(model(inputs))
            logits_ood = torch.cat(logits_ood, dim=0).cpu()

            # Confidence- and entropy-Scores of out of domain logits
            probas_ood = logits_ood.softmax(-1)
            conf_ood, _ = probas_ood.max(-1)
            entropy_ood = ood.entropy_fn(probas_ood)

            # Area under the Precision-Recall-Curve
            entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
            conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

            # Area under the Receiver-Operator-Characteristic-Curve
            entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
            conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

            # Add to metrics
            metrics[name+"_entropy_auroc"] = entropy_auroc
            metrics[name+"_conf_auroc"] = conf_auroc
            metrics[name+"_entropy_aupr"] = entropy_aupr
            metrics[name+"_conf_aupr"] = conf_aupr

        test_stats = {f"test_{k}": v for k, v in metrics.items()}

        return test_stats

    def write_scalar_dict(self, scalar_dict, prefix, global_step):
        if self.summary_writer is not None:
            for key, val in scalar_dict.items():
                self.summary_writer.add_scalar(f'{prefix}/{key}', val, global_step=global_step)
