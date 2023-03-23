import torch
import torch.nn.functional as F

from ..utils.trainer import BasicTrainer
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
    def __init__(self, model, optimizer, mixup_alpha, criterion, lr_scheduler=None, device=None, output_dir=None, summary_writer=None, use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.mixup_alpha = mixup_alpha

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

            inputs, targets = self.mixup(inputs, F.one_hot(targets), self.mixup_alpha)

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

    def mixup(self, inputs: torch.Tensor, targets: torch.Tensor, alpha: float):
        # TODO: as own function in utils
        indices = torch.randperm(len(inputs), device=inputs.device, dtype=torch.long)
        inputs_mixed = alpha * inputs + (1 - alpha) * inputs[indices]
        targets_mixed = alpha * targets + (1 - alpha) * targets[indices]
        return inputs_mixed, targets_mixed
