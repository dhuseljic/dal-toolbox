import torch

from ..utils.trainer import BasicTrainer
from ...utils import MetricLogger, SmoothedValue
from ...metrics import generalization, calibration, ood
from ..utils.variational_inference import KLCriterion


class VITrainer(BasicTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 mc_samples=10,
                 grad_norm=5,
                 kl_temperature=1,
                 kl_reduction='mean',
                 lr_scheduler=None,
                 device=None,
                 output_dir=None,
                 summary_writer=None,
                 use_distributed=False):
        super().__init__(model, optimizer, criterion, lr_scheduler, device, output_dir, summary_writer, use_distributed)
        self.kl_temperature = kl_temperature
        self.kl_reduction = kl_reduction
        self.kl_criterion = KLCriterion(reduction=kl_reduction)
        self.grad_norm = grad_norm
        self.mc_samples = mc_samples

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
            batch_size = inputs.shape[0]

            logits = self.model(inputs)

            nll = self.criterion(logits, targets)
            kl_weight = batch_size / len(dataloader.dataset)
            kl_loss = self.kl_temperature * kl_weight * self.kl_criterion(self.model)
            loss = nll + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

            acc1, = generalization.accuracy(logits, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), nll=nll.item(), kl_loss=kl_loss.item(),
                                 lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        metric_logger.synchronize_between_processes()
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

        return train_stats

    @torch.no_grad()
    def evaluate_model(self, dataloader, dataloaders_ood=None):
        self.model.eval()
        self.model.to(self.device)

        # Forward prop in distribution
        mc_logits = []
        all_targets = []
        for i_sample in range(self.mc_samples):
            logits = []
            for inputs, tar in dataloader:
                if i_sample == 0:
                    all_targets.append(tar)
                logits.append(self.model(inputs.to(self.device)).cpu())
            logits = torch.cat(logits)
            mc_logits.append(logits)
        mc_logits = torch.stack(mc_logits, dim=1)
        targets_id = torch.cat(all_targets)
        probas_id = mc_logits.softmax(dim=-1).mean(dim=1)

        # Model specific test loss and accuracy for in domain testset
        loss = (self.criterion(logits, targets_id) + self.kl_temperature * self.kl_criterion(self.model)).item()
        acc1 = generalization.accuracy(probas_id, targets_id, (1,))[0].item()
        nll = calibration.EnsembleCrossEntropy()(mc_logits, targets_id).item()
        gibbs_cross_entropy = calibration.GibbsCrossEntropy()(mc_logits, targets_id).item()
        brier = calibration.BrierScore()(probas_id, targets_id).item()
        tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
        mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

        metrics = {
            "loss": loss,
            "acc1": acc1,
            "nll": nll,
            "gibbs_cross_entropy": gibbs_cross_entropy,
            "brier": brier,
            "tce": tce,
            "mce": mce
        }

        if dataloaders_ood is None:
            dataloaders_ood = {}

        # for name, dataloader_ood in dataloaders_ood.items():
        #     # Forward prop out of distribution
        #     logits_ood, _ = self.collect_predictions(dataloader_ood)
        #     probas_ood = logits_ood.softmax(-1)

        #     # Confidence- and entropy-Scores of out of domain logits
        #     entropy_id = ood.entropy_fn(probas_id)
        #     entropy_ood = ood.entropy_fn(probas_ood)

        #     # Area under the Precision-Recall-Curve
        #     ood_aupr = ood.ood_aupr(entropy_id, entropy_ood)

        #     # Area under the Receiver-Operator-Characteristic-Curve
        #     ood_auroc = ood.ood_auroc(entropy_id, entropy_ood)

        #     # Add to metrics
        #     metrics[name+"_auroc"] = ood_auroc
        #     metrics[name+"_aupr"] = ood_aupr

        test_stats = {f"test_{k}": v for k, v in metrics.items()}
        return test_stats
