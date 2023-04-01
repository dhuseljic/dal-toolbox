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

        # Transform to tensor
        dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
        targets_id = torch.cat(targets_id, dim=0).cpu()

        # Transform into probabilitys
        dropout_probas_id = dropout_logits_id.softmax(dim=-1)

        # Average of probas per sample
        mean_probas_id = torch.mean(dropout_probas_id, dim=1)
        mean_probas_id = ood.clamp_probas(mean_probas_id)

        # Confidence- and entropy-Scores of in domain set logits
        conf_id, _ = mean_probas_id.max(-1)
        entropy_id = ood.entropy_fn(mean_probas_id)

        # Model specific test loss and accuracy for in domain testset
        acc1 = generalization.accuracy(torch.log(mean_probas_id), targets_id, (1,))[0].item()
        prec = generalization.avg_precision(mean_probas_id, targets_id)
        loss = self.criterion(torch.log(mean_probas_id), targets_id).item()

        # Negative Log Likelihood
        nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()
        brier = calibration.BrierScore()(mean_probas_id, targets_id).item()


        # Top- and Marginal Calibration Error
        tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
        mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

        metrics = {
            "acc1": acc1,
            "prec": prec,
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
                dropout_probas_ood = dropout_logits_ood.softmax(dim=-1)
                mean_probas_ood = torch.mean(dropout_probas_ood, dim=1)
                mean_probas_ood = ood.clamp_probas(mean_probas_ood)

                # Confidence- and entropy-Scores of out of domain logits
                conf_ood, _ = mean_probas_ood.max(-1)
                entropy_ood = ood.entropy_fn(mean_probas_ood)

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

        return {f"test_{k}": v for k, v in metrics.items()}
