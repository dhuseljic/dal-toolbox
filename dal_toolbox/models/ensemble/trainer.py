import torch

from ..utils.trainer import BasicTrainer
from ...utils import MetricLogger, SmoothedValue
from ... import metrics


class EnsembleTrainer(BasicTrainer):

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
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
        # Remove old model
        del self.model

        # Setup models
        model.members = torch.nn.ModuleList([self.fabric.setup(m) for m in model.members])
        self.model = model

        # Setup optimizer
        optimizer.optimizers = self.fabric.setup_optimizers(*optimizer.optimizers)
        self.optimizer = optimizer

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        train_stats = {}
        self.model.train()
        acc_fn = metrics.Accuracy().to(self.fabric.device)

        for i_member, (member, optim) in enumerate(zip(self.model, self.optimizer)):
            metric_logger = MetricLogger(delimiter=" ")
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            header = f"Epoch [{epoch}] Model [{i_member}] " if epoch is not None else f"Model [{i_member}] "

            # Train the epoch
            for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
                self.fabric.call("on_train_batch_start", self, member)

                outputs = member(inputs)

                loss = self.criterion(outputs, targets)

                optim.zero_grad()
                self.backward(loss)
                optim.step()

                batch_size = inputs.shape[0]
                acc1 = acc_fn(outputs, targets)
                metric_logger.update(loss=loss.item(), lr=optim.param_groups[0]["lr"])
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

                self.fabric.call("on_train_batch_end", self, member)
            metric_logger.synchronize_between_processes()
            train_stats.update({f"train_{key}_member{i_member}": meter.global_avg for key,
                                meter, in metric_logger.meters.items()})
        self.step_scheduler()
        return train_stats

    @torch.inference_mode()
    def predict(self, dataloader):
        self.model.eval()
        dataloader = self.fabric.setup_dataloaders(dataloader)

        logits_list = []
        targets_list = []
        for inputs, targets in dataloader:

            logits = self.model.forward_sample(inputs.to(self.device)).cpu()
            logits = self.all_gather(logits)
            targets = self.all_gather(targets)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)

        return logits, targets

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()

        ensemble_logits, targets = self.predict(dataloader)
        log_probas = metrics.ensemble_log_softmax(ensemble_logits)

        test_stats = {
            "accuracy": metrics.Accuracy()(log_probas, targets).item(),
            "loss": metrics.GibbsCrossEntropy()(ensemble_logits, targets).item(),
            "nll": metrics.EnsembleCrossEntropy()(ensemble_logits, targets).item(),
            "brier": metrics.BrierScore()(log_probas.exp(), targets).item(),
            "tce": metrics.ExpectedCalibrationError()(log_probas.exp(), targets).item(),
            "ace": metrics.AdaptiveCalibrationError()(log_probas.exp(), targets).item()
        }

        if dataloaders_ood is None:
            return test_stats
        for ds_name, dataloader_ood in dataloaders_ood.items():
            ensemble_logits_ood, _ = self.predict(dataloader_ood)

            entropy_id = metrics.ensemble_entropy_from_logits(ensemble_logits)
            entropy_ood = metrics.ensemble_entropy_from_logits(ensemble_logits_ood)

            aupr = metrics.ood_aupr(entropy_id, entropy_ood)
            auroc = metrics.ood_auroc(entropy_id, entropy_ood)

            test_stats[f"aupr_{ds_name}"] = aupr
            test_stats[f"auroc_{ds_name}"] = auroc

        return test_stats
