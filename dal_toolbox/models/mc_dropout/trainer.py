import torch

from ..utils.trainer import BasicTrainer
from ...metrics import generalization, calibration, ood
from ...utils import MetricLogger, SmoothedValue


class MCDropoutTrainer(BasicTrainer):
    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "
        self.model.to(self.device)
        self.model.train()

        for X_batch, y_batch in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            out = self.model(X_batch)
            loss = self.criterion(out, y_batch)
            batch_size = X_batch.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc1, = generalization.accuracy(out.softmax(dim=-1), y_batch, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()
        self.model.to(self.device)

        # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
        dropout_logits_id, targets_id, = [], []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            dropout_logits_id.append(self.model.mc_forward(inputs))
            targets_id.append(targets)
        dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
        targets_id = torch.cat(targets_id, dim=0).cpu()
        mean_probas_id = dropout_logits_id.softmax(dim=-1).mean(dim=1)

        acc1 = generalization.accuracy(mean_probas_id, targets_id, (1,))[0].item()
        loss = calibration.GibbsCrossEntropy()(dropout_logits_id, targets_id).item()
        nll = calibration.EnsembleCrossEntropy()(dropout_logits_id, targets_id).item()
        brier = calibration.BrierScore()(mean_probas_id, targets_id).item()
        tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
        mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

        metrics = {
            "acc1": acc1,
            "loss": loss,
            "nll": nll,
            "brier": brier,
            "tce": tce,
            "mce": mce
        }

        if dataloaders_ood:
            for name, dataloader_ood in dataloaders_ood.items():

                # Forward prop out of distribution
                dropout_logits_ood = []
                for inputs, targets in dataloader_ood:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    dropout_logits_ood.append(self.model.mc_forward(inputs))
                dropout_logits_ood = torch.cat(dropout_logits_ood, dim=0).cpu()

                # Compute entropy scores
                entropy_id = ood.ensemble_entropy_from_logits(dropout_logits_id)
                entropy_ood = ood.ensemble_entropy_from_logits(dropout_logits_ood)

                aupr = ood.ood_aupr(entropy_id, entropy_ood)
                auroc = ood.ood_auroc(entropy_id, entropy_ood)

                # Add to metrics
                metrics[name+"_auroc"] = auroc
                metrics[name+"_aupr"] = aupr

        return {f"test_{k}": v for k, v in metrics.items()}

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
