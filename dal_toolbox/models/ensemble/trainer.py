import torch

from ..utils.trainer import BasicTrainer
from ...utils import MetricLogger
from ...metrics import generalization, calibration, ood


class EnsembleTrainer(BasicTrainer):

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        train_stats = {}
        self.model.train()
        self.model.to(self.device)

        for i_member, (member, optim) in enumerate(zip(self.model, self.optimizer)):
            metric_logger = MetricLogger(delimiter=" ")
            header = f"Epoch [{epoch}] Model [{i_member}] " if epoch is not None else f"Model [{i_member}] "

            # Train the epoch
            for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = member(inputs)

                loss = self.criterion(outputs, targets)

                optim.zero_grad()
                loss.backward()
                optim.step()

                batch_size = inputs.shape[0]
                acc1, = generalization.accuracy(outputs, targets, topk=(1,))
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            train_stats.update({f"train_member{i_member}_{k}": meter.global_avg for k,
                                meter, in metric_logger.meters.items()})
        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()
        self.model.to(self.device)

        # Get logits and targets for in-domain-test-set (Number of Members x Number of Samples x Number of Classes)
        ensemble_logits_id, targets_id, = [], []
        for inputs, targets in dataloader:
            logits = self.model.forward_sample(inputs.to(self.device)).cpu()
            ensemble_logits_id.append(logits)
            targets_id.append(targets)

        # Transform to tensor
        ensemble_logits_id = torch.cat(ensemble_logits_id, dim=0)
        targets_id = torch.cat(targets_id, dim=0)

        # Transform into probabilitys
        ensemble_probas_id = ensemble_logits_id.softmax(dim=-1)

        # Average of probas per sample
        mean_probas_id = torch.mean(ensemble_probas_id, dim=1)

        # Confidence- and entropy-Scores of in domain set logits
        conf_id, _ = mean_probas_id.max(-1)
        entropy_id = ood.entropy_fn(mean_probas_id)

        # Compute accuracy
        acc1 = generalization.accuracy(mean_probas_id, targets_id, (1,))[0].item()
        # Avg cross entropy of all members
        loss = calibration.GibsCrossEntropy()(ensemble_logits_id, targets_id).item()

        # Cross entropy of ensemble using the predictive distribution
        nll = calibration.EnsembleCrossEntropy()(ensemble_logits_id, targets_id).item()
        brier = calibration.BrierScore()(mean_probas_id, targets_id).item()

        # Top- and Marginal Calibration Error
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

        if dataloaders_ood is None:
            dataloaders_ood = {}

        for name, dataloader_ood in dataloaders_ood.items():
            # Repeat for out-of-domain-test-set
            ensemble_logits_ood = []
            for inputs, targets in dataloader_ood:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                ensemble_logits_ood.append(self.model.forward_sample(inputs))
            ensemble_logits_ood = torch.cat(ensemble_logits_ood, dim=1).cpu()
            ensemble_probas_ood = ensemble_logits_ood.softmax(dim=-1)
            mean_probas_ood = torch.mean(ensemble_probas_ood, dim=0)

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
