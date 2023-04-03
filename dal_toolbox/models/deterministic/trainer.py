import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ..utils.ssl_utils import FlexMatchThresholdingHook
from ..utils.trainer import BasicTrainer
from ..utils import unfreeze_bn, freeze_bn
from ...metrics import generalization, calibration, ood
from ...utils import MetricLogger, SmoothedValue


class DeterministicTrainer(BasicTrainer):

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()
        self.model.to(self.device)

        # Forward prop in distribution
        logits_id, targets_id, = [], []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits_id.append(self.model(inputs))
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

        if dataloaders_ood is None:
            dataloaders_ood = {}

        for name, dataloader_ood in dataloaders_ood.items():
            # Forward prop out of distribution
            logits_ood = []
            for inputs, targets in dataloader_ood:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits_ood.append(self.model(inputs))
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


class DeterministicMixupTrainer(DeterministicTrainer):
    def __init__(self, model, criterion, mixup_alpha, n_classes, optimizer, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.mixup_alpha = mixup_alpha
        self.n_classes = n_classes

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            inputs, targets = self.mixup(inputs, F.one_hot(targets, num_classes=self.n_classes), self.mixup_alpha)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

        return train_stats

    def mixup(self, inputs: torch.Tensor, targets_one_hot: torch.Tensor, alpha: float):
        # TODO: move to utils
        indices = torch.randperm(len(inputs), device=inputs.device, dtype=torch.long)
        inputs_mixed = alpha * inputs + (1 - alpha) * inputs[indices]
        targets_mixed = alpha * targets_one_hot + (1 - alpha) * targets_one_hot[indices]
        return inputs_mixed, targets_mixed
    


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

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
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
            loss = loss_l + self.unsup_warmup * self.lambda_u * loss_u

            # Backpropagation and Lr-Scheduler-Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            batch_size_l, batch_size_u = x_l.shape[0], x_u.shape[0]
            acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
            pseudo_acc1, = generalization.accuracy(logits_u, y_u, topk=(1,))
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
        self.unsup_warmup =  unsup_warmup

    def train_one_epoch(self, labeled_loader, unlabeled_loader_weak_1, unlabeled_loader_weak_2, epoch=None, print_freq=200):
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        unlabeled_iter1 = iter(unlabeled_loader_weak_1)
        unlabeled_iter2 = iter(unlabeled_loader_weak_2)

        i_iter = epoch*len(labeled_loader)
        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            (x_w1, a), (x_w2, b) = next(unlabeled_iter1), next(unlabeled_iter2)
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
            acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
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

    def train_one_epoch(self, labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, epoch=None, print_freq=200):
        self.model.to(self.device)
        self.model.train()
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        iterator_unlabeled_weak_aug = iter(unlabeled_loader_weak)
        iterator_unlabeled_strong_aug = iter(unlabeled_loader_strong)

        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            (x_w, y_w), (x_s, _)  = next(iterator_unlabeled_weak_aug), next(iterator_unlabeled_strong_aug)
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)
            x_w = x_w.to(self.device)
            x_s = x_s.to(self.device)
            y_w = y_w.to(self.device)

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
            acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
            pseudo_acc1, = generalization.accuracy(logits_w, y_w, topk=(1,))
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

    def train_one_epoch(self, labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, unlabeled_loader_indices, epoch=None, print_freq=200):
        self.model.to(self.device)
        self.model.train()
        self.criterion.to(self.device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        iterator_unlabeled_weak_aug = iter(unlabeled_loader_weak)
        iterator_unlabeled_strong_aug = iter(unlabeled_loader_strong)
        iterator_unlabeled_indices = iter(unlabeled_loader_indices)

        for x_l, y_l in metric_logger.log_every(labeled_loader, print_freq=print_freq, header=header):
            (x_w, y_w), (x_s, _), idx  = next(iterator_unlabeled_weak_aug), next(iterator_unlabeled_strong_aug), next(iterator_unlabeled_indices)
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device)
            x_w = x_w.to(self.device)
            x_s = x_s.to(self.device)
            y_w = y_w.to(self.device)
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
            acc1, = generalization.accuracy(logits_l, y_l, topk=(1,))
            pseudo_acc1, = generalization.accuracy(logits_w, y_w, topk=(1,))
            metric_logger.update(loss=loss.item(), supervised_loss=loss_l.item(), lr=self.optimizer.param_groups[0]["lr"],
                                unsupervised_loss=loss_u.item(), mask_ratio=mask.float().mean().item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["pseudo_acc1"].update(pseudo_acc1.item(), n=ulb_batch_size)
        
        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats