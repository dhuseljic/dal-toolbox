import torch

from ..deterministic.trainer import DeterministicTrainer
from ..utils.random_features import RandomFeatureGaussianProcess
from ...utils import MetricLogger, SmoothedValue
from ...metrics import generalization, calibration, ood


class SNGPTrainer(DeterministicTrainer):

    def _reset_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.reset_precision_matrix()

    def _synchronize_precision_matrix(self):
        for m in self.model.modules():
            if isinstance(m, RandomFeatureGaussianProcess):
                m.synchronize_precision_matrix()

    def train_one_epoch(self, dataloader, epoch=None, print_freq=200):
        self.model.train()
        self._reset_precision_matrix()

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
            acc1, = generalization.accuracy(logits, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

            self.fabric.call("on_train_batch_end", self, self.model)
        self.step_scheduler()
        self._synchronize_precision_matrix()
        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
        return train_stats

    @torch.inference_mode()
    def predict(self, dataloader):
        self.model.eval()
        dataloader = self.fabric.setup_dataloaders(dataloader)

        logits_list = []
        targets_list = []
        for inputs, targets in dataloader:
            logits = self.model(inputs, mean_field=True)

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
        self.model.to(self.device)

        # Forward prop in distribution
        logits_id, targets_id = [], []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits_scaled = self.model(inputs, mean_field=True)
            logits_id.append(logits_scaled)
            targets_id.append(targets)
        logits_id = torch.cat(logits_id, dim=0).cpu()
        targets_id = torch.cat(targets_id, dim=0).cpu()
        probas_id = logits_id.softmax(-1)

        # Model specific test loss and accuracy for in domain testset
        acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
        loss = self.criterion(logits_id, targets_id).item()
        nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()
        tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
        mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

        metrics = {
            "acc1": acc1,
            "loss": loss,
            "nll": nll,
            "tce": tce,
            "mce": mce
        }

        # TODO
        conf_id, _ = probas_id.max(-1)
        entropy_id = ood.entropy_fn(probas_id)
        dempster_shafer_id = ood.dempster_shafer_uncertainty(logits_id)

        if dataloaders_ood is None:
            dataloaders_ood = {}

        for name, dataloader_ood in dataloaders_ood.items():
            # Forward prop out of distribution
            logits_ood = []
            for inputs, targets in dataloader_ood:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits_scaled = self.model(inputs, mean_field=True)
                logits_ood.append(logits_scaled)
            logits_ood = torch.cat(logits_ood, dim=0).cpu()

            # Confidence- and entropy-Scores of out of domain logits
            probas_ood = logits_ood.softmax(-1)
            conf_ood, _ = probas_ood.max(-1)
            entropy_ood = ood.entropy_fn(probas_ood)
            dempster_shafer_ood = ood.dempster_shafer_uncertainty(logits_ood)

            # Area under the Precision-Recall-Curve
            entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
            conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)
            dempster_shafer_aupr = ood.ood_aupr(dempster_shafer_id, dempster_shafer_ood)

            metrics[name+"_entropy_aupr"] = entropy_aupr
            metrics[name+"_conf_aupr"] = conf_aupr
            metrics[name+"_dempster_aupr"] = dempster_shafer_aupr

            # Area under the Receiver-Operator-Characteristic-Curve
            entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
            conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)
            dempster_shafer_auroc = ood.ood_aupr(dempster_shafer_id, dempster_shafer_ood)

            metrics[name+"_entropy_auroc"] = entropy_auroc
            metrics[name+"_conf_auroc"] = conf_auroc
            metrics[name+"_dempster_auroc"] = dempster_shafer_auroc

        return {f"test_{k}": v for k, v in metrics.items()}
