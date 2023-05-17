import torch

from ..utils.trainer import BasicTrainer
from ...utils import MetricLogger, SmoothedValue
from ... import metrics


class MCDropoutTrainer(BasicTrainer):
    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

        for inputs, targets in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
            self.fabric.call("on_train_batch_start", self, self.model)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            batch_size = inputs.size(0)

            self.optimizer.zero_grad()
            self.backward(loss)
            self.optimizer.step()

            acc1 = acc_fn(logits, targets)
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            self.fabric.call("on_train_batch_end", self, self.model)

        self.step_scheduler()
        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()

        # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
        dropout_logits, targets = self.predict(dataloader)
        log_probas = metrics.ensemble_log_softmax(dropout_logits)

        test_stats = {
            "accuracy": metrics.Accuracy()(log_probas, targets).item(),
            "loss": metrics.GibbsCrossEntropy()(dropout_logits, targets).item(),
            "nll": metrics.EnsembleCrossEntropy()(dropout_logits, targets).item(),
            "brier": metrics.BrierScore()(log_probas.exp(), targets).item(),
            "tce": metrics.ExpectedCalibrationError()(log_probas.exp(), targets).item(),
            "ace": metrics.AdaptiveCalibrationError()(log_probas.exp(), targets).item()
        }

        if dataloaders_ood is None:
            return test_stats
        for ds_name, dataloader_ood in dataloaders_ood.items():
            dropout_logits_ood, _ = self.predict(dataloader_ood)

            # Compute entropy scores
            entropy_id = metrics.ensemble_entropy_from_logits(dropout_logits)
            entropy_ood = metrics.ensemble_entropy_from_logits(dropout_logits_ood)

            aupr = metrics.OODAUPR()(entropy_id, entropy_ood).item()
            auroc = metrics.OODAUROC()(entropy_id, entropy_ood).item()

            test_stats.update({
                f"aupr_{ds_name}": aupr,
                f"auroc_{ds_name}": auroc,
            })
        return test_stats

    @torch.inference_mode()
    def predict(self, dataloader):
        self.model.eval()
        # self.model.to(self.device)
        dataloader = self.fabric.setup_dataloaders(dataloader)

        logits_list = []
        targets_list = []
        for inputs, targets in dataloader:
            # inputs = inputs.to(self.device)
            # targets = targets.to(self.device)
            logits = self.model.mc_forward(inputs)

            logits = self.all_gather(logits)
            targets = self.all_gather(targets)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)

        return logits, targets
