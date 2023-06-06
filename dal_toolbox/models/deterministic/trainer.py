import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ..utils import unfreeze_bn, freeze_bn
from ..utils.ssl_utils import FlexMatchThresholdingHook
from ..utils.trainer import BasicTrainer
from ..utils.mixup import mixup
from ... import metrics
from ...utils import MetricLogger, SmoothedValue


class DeterministicTrainer(BasicTrainer):

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            self.fabric.call("on_train_batch_start", self, self.model)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            self.backward(loss)
            self.optimizer.step()

            batch_size = inputs.shape[0]
            acc1 = acc_fn(logits, targets)
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1, n=batch_size)

            self.fabric.call("on_train_batch_end", self, self.model)
        self.step_scheduler()

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()
        self.model.to(self.device)

        # Forward prop in distribution
        logits_id, targets_id = self.predict(dataloader)
        probas_id = logits_id.softmax(-1)

        # Model specific test loss and accuracy for in domain testset
        test_stats = {
            "loss": self.criterion(logits_id, targets_id).item(),
            "accuracy": metrics.Accuracy()(logits_id, targets_id).item(),
            "nll": torch.nn.CrossEntropyLoss()(logits_id, targets_id).item(),
            "brier": metrics.BrierScore()(probas_id, targets_id).item(),
            "tce": metrics.ExpectedCalibrationError()(probas_id, targets_id).item(),
            "ace": metrics.AdaptiveCalibrationError()(probas_id, targets_id).item(),
        }
        if dataloaders_ood is None:
            return test_stats

        for ds_name, dataloader_ood in dataloaders_ood.items():
            # Forward prop out of distribution
            logits_ood, _ = self.predict(dataloader_ood)
            entropy_id = metrics.entropy_from_logits(logits_id)
            entropy_ood = metrics.entropy_from_logits(logits_ood)

            ood_aupr = metrics.OODAUPR()(entropy_id, entropy_ood).item()
            ood_auroc = metrics.OODAUROC()(entropy_id, entropy_ood).item()

            # Add to metrics
            test_stats[f"aupr_{ds_name}"] = ood_aupr
            test_stats[f"auroc_{ds_name}"] = ood_auroc
        return test_stats

    @torch.inference_mode()
    def predict(self, dataloader):
        self.model.eval()
        dataloader = self.fabric.setup_dataloaders(dataloader)

        logits_list = []
        targets_list = []
        for inputs, targets in dataloader:
            logits = self.model(inputs)

            logits = self.all_gather(logits)
            targets = self.all_gather(targets)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)

        return logits, targets


class DeterministicMixupTrainer(DeterministicTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        mixup_alpha: float,
        lr_scheduler=None,
        num_epochs: int = 200,
        callbacks: list = [],
        device: str = None,
        num_devices: int = 'auto',
        precision='32-true',
        output_dir: str = None,
        summary_writer=None,
        val_every: int = 1,
        save_every: int = None,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler, num_epochs, callbacks,
                         device, num_devices, precision, output_dir, summary_writer, val_every, save_every)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            self.fabric.call("on_train_batch_start", self, self.model)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
            inputs, targets = mixup(inputs, targets_one_hot, mixup_alpha=self.mixup_alpha)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            self.backward(loss)
            self.optimizer.step()

            batch_size = inputs.shape[0]
            acc1 = acc_fn(logits, targets)
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            self.fabric.call("on_train_batch_end", self, self.model)

        self.step_scheduler()
        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats


class DeterministicPseudoLabelTrainer(DeterministicTrainer):
    def __init__(self, model, criterion, n_classes, optimizer, n_iter, p_cutoff, unsup_warmup, lambda_u, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.n_classes = n_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.n_iter = n_iter
        self.lambda_u = lambda_u
        self.p_cutoff = p_cutoff
        self.unsup_warmup = unsup_warmup

    def train_one_epoch(self, labeled_loader, unlabeled_loader, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        unlabeled_iter = iter(unlabeled_loader)

        i_iter = epoch*len(labeled_loader)
        for (x_l, y_l) in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            x_u, y_u = next(unlabeled_iter)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u, y_u = x_u.to(self.device), y_u.to(self.device)

            # Get all necesseracy model outputs
            logits_l = self.model(x_l)
            bn_backup = freeze_bn(self.model)
            logits_u = self.model(x_u)
            unfreeze_bn(self.model, bn_backup)

            # Generate pseudo labels and mask
            probas_ulb = torch.softmax(logits_u.detach(), dim=-1)
            max_probas, pseudo_label = torch.max(probas_ulb, dim=-1)
            mask = max_probas.ge(self.p_cutoff)

            # Warm Up Factor
            unsup_warmup_factor = np.clip(i_iter / (self.unsup_warmup*self.n_iter), a_min=0, a_max=1)
            i_iter += 1

            # Calculate Loss
            loss_l = self.criterion(logits_l, y_l)
            loss_u = torch.mean(self.ce_loss(logits_u, pseudo_label) * mask)
            loss = loss_l + self.unsup_warmup * self.lambda_u * loss_u # BUG: not using unsup warmup factor

            # Backpropagation and Lr-Scheduler-Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            batch_size_l, batch_size_u = x_l.shape[0], x_u.shape[0]
            acc1 = acc_fn(logits_l, y_l)
            pseudo_acc1 = acc_fn(logits_u, y_u)
            metric_logger.update(loss=loss.item(), sup_loss=loss_l.item(), unsup_loss=loss_u.item(),
                                 mask_ratio=mask.float().mean().item(), unsup_warmup_factor=unsup_warmup_factor, lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size_l)
            metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=batch_size_u)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats


class DeterministicPiModelTrainer(DeterministicTrainer):
    def __init__(self, model, criterion, n_classes, optimizer, n_iter, lambda_u, unsup_warmup, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.n_classes = n_classes
        self.lambda_u = lambda_u
        self.n_iter = n_iter
        self.unsup_warmup = unsup_warmup

    def train_one_epoch(self, labeled_loader, unlabeled_loader, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        unlabeled_iter = iter(unlabeled_loader)

        i_iter = epoch*len(labeled_loader)
        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            (x_w1, x_w2, t) = next(unlabeled_iter)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_w1 = x_w1.to(self.device)
            x_w2 = x_w2.to(self.device)

            # Get all necesseracy model outputs
            logits_l = self.model(x_l)
            bn_backup = freeze_bn(self.model)
            logits_w1 = self.model(x_w1)
            logits_w2 = self.model(x_w2)
            unfreeze_bn(self.model, bn_backup)

            # Warm Up Factor
            unsup_warmup_factor = np.clip(i_iter / (self.unsup_warmup*self.n_iter), a_min=0, a_max=1)
            i_iter += 1

            # Calculate Loss
            loss_l = self.criterion(logits_l, y_l)
            loss_u = F.mse_loss(logits_w2.softmax(-1), logits_w1.detach().softmax(-1))
            loss = loss_l + unsup_warmup_factor * self.lambda_u * loss_u

            # Backpropagation and Lr-Scheduler-Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            batch_size = x_l.shape[0]
            acc1 = acc_fn(logits_l, y_l)
            metric_logger.update(loss=loss.item(), sup_loss=loss_l.item(), unsup_loss=loss_u.item(),
                                 unsup_warmup_factor=unsup_warmup_factor, lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats


class DeterministicFixMatchTrainer(DeterministicTrainer):
    def __init__(self, model, criterion, n_classes, optimizer, n_iter, lambda_u, p_cutoff, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.n_classes = n_classes
        self.lambda_u = lambda_u
        self.n_iter = n_iter
        self.p_cutoff = p_cutoff
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def train_one_epoch(self, labeled_loader, unlabeled_loader, epoch=None, print_freq=200):
        self.model.to(self.device)
        self.model.train()
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            x_w, x_s, y = next(unlabeled_iter)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_w, y = x_w.to(self.device), y.to(self.device)
            x_s = x_s.to(self.device)

            # Get all necesseracy model outputs
            logits_l = self.model(x_l)
            with torch.no_grad():
                logits_w = self.model(x_w)
            logits_s = self.model(x_s)

            # Calculate pseudolabels and mask
            probas_w = torch.softmax(logits_w, dim=-1)
            y_probs, y_ps = probas_w.max(-1)
            mask = y_probs.ge(self.p_cutoff).to(self.device)

            # Calculate Loss
            loss_l = self.criterion(logits_l, y_l)
            loss_u = torch.mean(self.ce_loss(logits_s, y_ps) * mask)
            loss = loss_l + self.lambda_u * loss_u

            # Backpropagation and Lr-Scheduler-Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            batch_size, ulb_batch_size = x_l.shape[0], x_s.shape[0]
            acc1 = metrics.Accuracy()(logits_l, y_l)
            pseudo_acc1 = metrics.Accuracy()(logits_w, y)
            metric_logger.update(loss=loss.item(), supervised_loss=loss_l.item(), lr=self.optimizer.param_groups[0]["lr"],
                                 unsupervised_loss=loss_u.item(), mask_ratio=mask.float().mean().item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=ulb_batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats


class DeterministicFlexMatchTrainer(DeterministicTrainer):
    def __init__(self, model, criterion, n_classes, optimizer, n_iter, lambda_u, p_cutoff, ulb_ds_len, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.n_classes = n_classes
        self.lambda_u = lambda_u
        self.n_iter = n_iter
        self.p_cutoff = p_cutoff
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.fmth = FlexMatchThresholdingHook(ulb_dest_len=ulb_ds_len, num_classes=n_classes, thresh_warmup=True)

    def train_one_epoch(self, labeled_loader, unlabeled_loader, epoch=None, print_freq=200):
        self.model.to(self.device)
        self.model.train()
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        unlabeled_iter = iter(unlabeled_loader)

        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            x_w, x_s, y, idx = next(unlabeled_iter)
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)
            x_w = x_w.to(self.device)
            x_s = x_s.to(self.device)
            y = y.to(self.device)
            idx = idx.to(self.device)

            # Get all necesseracy model outputs
            logits_l = self.model(x_l)
            with torch.no_grad():
                logits_w = self.model(x_w)
            logits_s = self.model(x_s)

            # Calculate pseudolabels and mask
            probas_w = torch.softmax(logits_w, dim=-1)
            _, y_ps = probas_w.max(-1)
            mask = self.fmth.masking(self.p_cutoff, probas_w, idx)

            # Calculate Loss
            loss_l = self.criterion(logits_l, y_l)
            loss_u = torch.mean(self.ce_loss(logits_s, y_ps) * mask)
            loss = loss_l + self.lambda_u * loss_u

            # Backpropagation and Lr-Scheduler-Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            batch_size, ulb_batch_size = x_l.shape[0], x_s.shape[0]
            acc1 = metrics.Accuracy()(logits_l, y_l)
            pseudo_acc1 = metrics.Accuracy()(logits_w, y)
            metric_logger.update(loss=loss.item(), supervised_loss=loss_l.item(), lr=self.optimizer.param_groups[0]["lr"],
                                 unsupervised_loss=loss_u.item(), mask_ratio=mask.float().mean().item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=ulb_batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats
