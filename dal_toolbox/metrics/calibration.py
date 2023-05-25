import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

from sklearn import metrics
from .utils import log_sub_exp


class CrossEntropy(torchmetrics.Metric):
    """Standard cross entropy."""

    def __init__(self):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.add_state('cross_entropy_list', default=[], dist_reduce_fx='cat')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim != 2:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        ce = self.cross_entropy(logits, targets)
        self.cross_entropy_list.append(ce)

    def compute(self):
        cross_entropies = torch.cat(self.cross_entropy_list, dim=0)
        return torch.mean(cross_entropies)


class EnsembleCrossEntropy(torchmetrics.Metric):
    """Cross entropy for a ensemble of predictions.

    For each datapoint (x,y), the ensemble's negative log-probability is:

    ```
    -log p(y|x) = -log 1/ensemble_size sum_{m=1}^{ensemble_size} p(y|x,theta_m)
                = -log sum_{m=1}^{ensemble_size} exp(log p(y|x,theta_m)) + log ensemble_size.
    ```

    Reference:
        https://github.com/google-research/robustness_metrics/blob/master/robustness_metrics/metrics/information_criteria.py#L74-L89

    """

    def __init__(self):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.add_state('cross_entropy_list', default=[], dist_reduce_fx='cat')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        num_samples, ensemble_size, _ = logits.shape

        # Reshape logits from N x M x C -> N x C x M
        _logits = logits.permute(0, 2, 1)
        # Expand labels from N -> N x M
        _targets = targets.view(-1, 1).expand(num_samples, ensemble_size)
        ce = self.cross_entropy(_logits, _targets)
        ce = -torch.logsumexp(-ce, dim=-1) + math.log(ensemble_size)
        self.cross_entropy_list.append(ce)

    def compute(self):
        cross_entropies = torch.cat(self.cross_entropy_list, dim=0)
        return torch.mean(cross_entropies)


class GibbsCrossEntropy(torchmetrics.Metric):
    """Average cross entropy of ensemble members.

    For each datapoint (x,y), the ensemble's Gibbs cross entropy is:

    ```
    - (1/ensemble_size) sum_{m=1}^ensemble_size log p(y|x,theta_m).
    ```

    Reference:
        https://github.com/google-research/robustness_metrics/blob/master/robustness_metrics/metrics/information_criteria.py#L92-L155

    """

    def __init__(self):
        super().__init__()
        self.add_state('cross_entropy_per_sample', default=[], dist_reduce_fx='cat')
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        num_samples, ensemble_size, _ = logits.shape

        # Reshape logits from N x M x C -> N x C x M
        _logits = logits.permute(0, 2, 1)
        # Expand labels from N -> N x M
        _targets = targets.view(-1, 1).expand(num_samples, ensemble_size)
        ce = self.cross_entropy(_logits, _targets)
        ce = torch.mean(ce, dim=-1)
        self.cross_entropy_per_sample.append(ce)

    def compute(self):
        cross_entropy_per_sample = torch.cat(self.cross_entropy_per_sample, dim=0)
        return torch.mean(cross_entropy_per_sample)


class BrierScore(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('brier_score_list', default=[], dist_reduce_fx='cat')

    def update(self, logits, labels):
        num_samples, num_classes = logits.shape
        assert len(labels) == num_samples, "Probas and Labels must be of the same size"
        probas = logits.softmax(dim=-1)
        labels_onehot = F.one_hot(labels, num_classes=num_classes)
        score = torch.sum(F.mse_loss(probas, labels_onehot, reduction='none'), dim=-1)
        # Note: Tensorflow ignores the addition by one. To be consistent with the decomposition, we also ignore it.
        # probas_label = torch.sum(probas * labels_onehot, -1)
        # score = torch.sum(probas**2, dim=-1) - 2 * probas_label +1
        # score = torch.mean(score, -1)

        self.brier_score_list.append(score)

    def compute(self):
        brier_scores = torch.cat(self.brier_score_list, dim=0)
        return torch.mean(brier_scores)


class BrierScoreDecomposition(nn.Module):
    # TODO(dhuseljic): Transform to torchmetrics
    # From: https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/stats/calibration.py
    @torch.no_grad()
    def forward(self, logits, targets):
        n_samples, n_classes = logits.shape

        preds = logits.argmax(dim=-1)
        confusion_matrix = metrics.confusion_matrix(targets, preds, labels=range(n_classes)).T
        confusion_matrix = torch.from_numpy(confusion_matrix).float()

        # n_k
        dist_weights = torch.sum(confusion_matrix, dim=-1)
        dist_weights = dist_weights / torch.sum(dist_weights, dim=-1, keepdim=True)

        # o_k_bar
        pbar = torch.sum(confusion_matrix, dim=-2)
        pbar = pbar / torch.sum(pbar, dim=-1, keepdim=True)

        # o_bar
        eps = torch.finfo(confusion_matrix.dtype).eps
        dist_mean = confusion_matrix / (torch.sum(confusion_matrix, dim=-1, keepdim=True) + eps)

        uncertainty = - torch.sum(torch.square(pbar), dim=-1)

        resolution = torch.square(pbar.unsqueeze(-1) - dist_mean)
        resolution = torch.sum(dist_weights * torch.sum(resolution, dim=-1), dim=-1)

        # f_k
        prob_true = dist_mean[preds]

        log_prob_true = prob_true.log()
        log_prob_pred = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        log_reliability = torch.logsumexp(2 * log_sub_exp(log_prob_pred, log_prob_true), dim=-1)
        log_reliability = torch.logsumexp(log_reliability, dim=-1)
        reliabilty = torch.exp(log_reliability - math.log(n_samples))
        out = {
            'uncertainty': uncertainty.item(),
            'resolution': resolution.item(),
            'reliability': reliabilty.item(),
        }
        return out


def calibration_error(confs: torch.Tensor, accs: torch.Tensor, n_samples: torch.Tensor, p: int = 2):
    probas_bin = n_samples/n_samples.nansum()
    ce = (torch.nansum(probas_bin * torch.abs(confs-accs)**p))**(1/p)
    return ce


class TopLabelCalibrationPlot(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, num_bins=15):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.num_bins+1)

        confs = torch.Tensor(self.num_bins)
        accs = torch.Tensor(self.num_bins)
        n_samples = torch.Tensor(self.num_bins)

        pred_confs, pred_labels = probas.max(dim=-1)

        for i_bin, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (bin_start < pred_confs) & (pred_confs < bin_end)
            n_samples[i_bin] = in_bin.sum()

            if in_bin.sum() == 0:
                confs[i_bin] = float('nan')
                accs[i_bin] = float('nan')
                continue

            bin_conf = pred_confs[in_bin].mean()
            bin_acc = (pred_labels[in_bin] == labels[in_bin]).float().mean()

            confs[i_bin] = bin_conf
            accs[i_bin] = bin_acc

        self.results = {'confs': confs, 'accs': accs, 'n_samples': n_samples}
        # calibration_error(confs, accs, n_samples, self.p)
        return self.results


class TopLabelCalibrationError(TopLabelCalibrationPlot):
    def __init__(self, num_bins=15, p=1):
        super().__init__(num_bins)
        self.p = p

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        results = super().forward(probas, labels)
        confs = results['confs']
        accs = results['accs']
        n_samples = results['n_samples']
        return calibration_error(confs, accs, n_samples, self.p)


class MarginalCalibrationPlot(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, num_bins=15, threshold=0.01):
        super().__init__()
        self.num_bins = num_bins
        self.threshold = threshold
        self.results = []

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.num_bins+1)
        _, n_classes = probas.shape

        # Save calibration plots in results
        self.results = []
        for i_cls in range(n_classes):
            labels_cls = (labels == i_cls).long()
            probas_cls = probas[:, i_cls]

            labels_cls = labels_cls[probas_cls > self.threshold]
            probas_cls = probas_cls[probas_cls > self.threshold]

            confs = torch.Tensor(self.num_bins)
            accs = torch.Tensor(self.num_bins)
            n_samples = torch.Tensor(self.num_bins)
            for i_bin, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                in_bin = (bin_start < probas_cls) & (probas_cls < bin_end)
                n_samples[i_bin] = in_bin.sum()

                if in_bin.sum() == 0:
                    confs[i_bin] = float('nan')
                    accs[i_bin] = float('nan')
                    continue

                bin_conf = probas_cls[in_bin].mean()
                bin_acc = (labels_cls[in_bin] == 1).float().mean()

                confs[i_bin] = bin_conf
                accs[i_bin] = bin_acc
            self.results.append({'confs': confs, 'accs': accs, 'n_samples': n_samples, 'class': i_cls})

        # sq_ces = [calibration_error(d['confs'], d['accs'], d['n_samples'], self.p)**self.p for d in self.results]
        # mce = torch.Tensor(sq_ces).mean()**(1/self.p)
        return self.results


class MarginalCalibrationError(MarginalCalibrationPlot):
    def __init__(self, num_bins=15, threshold=0.01, p=1):
        super().__init__(num_bins, threshold)
        self.p = p

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        results = super().forward(probas, labels)
        sq_ces = [calibration_error(d['confs'], d['accs'], d['n_samples'], self.p)**self.p for d in results]
        return torch.Tensor(sq_ces).mean()**(1/self.p)


# From https://github.com/google-research/robustness_metrics/robustness_metrics/metrics/uncertainty.py#L1464-L1648
class GeneralCalibrationError(nn.Module):
    # Not in torchmetrics format, it is a fullbatch metric since binning depends on all predictions
    def __init__(self,
                 binning_scheme: str,
                 max_prob: bool,
                 class_conditional: bool,
                 norm: str,
                 num_bins: int,
                 threshold: float,
                 ):
        super().__init__()
        self.binning_scheme = binning_scheme
        self.max_prob = max_prob
        self.class_conditional = class_conditional
        self.norm = norm
        self.num_bins = num_bins
        self.threshold = threshold

    def _get_adaptive_bins(self, probas, num_bins):
        """Returns upper edges for binning an equal number of datapoints per bin."""
        if probas.numel() == 0:
            return torch.linspace(0, 1, steps=self.num_bins+1)[1:]
        probas = probas.view(-1)
        edge_indices = torch.linspace(0, probas.numel(), num_bins+1)[:-1]
        edge_indices = torch.round(edge_indices).long()
        edge_indices = torch.minimum(edge_indices, torch.tensor(probas.numel()-1))
        sorted_probas = torch.sort(probas).values
        edges = sorted_probas[edge_indices]
        upper_bounds = torch.cat([edges, torch.Tensor([1.])])
        return upper_bounds

    def _get_upper_bounds(self, probas, targets):
        if self.binning_scheme == 'even' and self.num_bins is not None:
            upper_bounds = torch.linspace(0, 1, steps=self.num_bins+1)[1:]
        elif self.binning_scheme == 'adaptive' and self.num_bins is not None:
            upper_bounds = self._get_adaptive_bins(probas, num_bins=self.num_bins)
        elif self.binning_scheme == "even" and self.num_bins is None:
            # TODO(dhuseljic): implement this
            # upper_bounds = get_mon_sweep_bins(probas, targets, binning_scheme=self.binning_scheme)
            raise NotImplementedError('Binning not implemented')
        elif self.binning_scheme == "adaptive" and self.num_bins is None:
            # TODO(dhuseljic): implement this
            # upper_bounds = get_mon_sweep_bins(probas, targets, binning_scheme=self.binning_scheme)
            raise NotImplementedError('Binning not implemented')
        else:
            raise NotImplementedError('Binning not implemented')
        return upper_bounds

    @torch.no_grad()
    def forward(self, logits, targets):
        num_classes = logits.shape[-1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        probas = F.softmax(logits, dim=-1)

        if not self.class_conditional:
            # only top pred
            if self.max_prob:
                preds = probas.argmax(-1)
                probas = probas[range(len(probas)), preds]
                targets_one_hot = targets_one_hot[range(len(probas)), preds]

            # Threshold
            targets = torch.squeeze(targets_one_hot[probas > self.threshold])
            probas = torch.squeeze(probas[probas > self.threshold])

            # Get Bounds
            upper_bounds = self._get_upper_bounds(probas, targets)

            calibration_error = self._compute_calibration_error(probas, targets, upper_bounds)
        else:
            calibration_error_list = []
            for k in range(num_classes):
                if not self.max_prob:
                    probas_class = probas[:, k]
                    targets_class = targets_one_hot[:, k]
                    targets_class = targets_class[probas_class > self.threshold]
                    probas_class = probas_class[probas_class > self.threshold]
                    upper_bounds = self._get_upper_bounds(probas_class, targets_class)

                    class_calibration_error = self._compute_calibration_error(probas_class, targets_class, upper_bounds)
                    calibration_error_list.append(class_calibration_error / num_classes)
                else:
                    preds = probas.argmax(-1)
                    targets_class = targets_one_hot[preds == k][:, k]
                    probas_class = probas[preds == k][:, k]

                    targets_class = targets_class[probas_class > self.threshold]
                    probas_class = probas_class[probas_class > self.threshold]

                    upper_bounds = self._get_upper_bounds(probas_class, targets_class)
                    class_calibration_error = self._compute_calibration_error(probas_class, targets_class, upper_bounds)
                    calibration_error_list.append(class_calibration_error / num_classes)
            calibration_error = torch.stack(calibration_error_list).sum()

        if self.norm == 'l2':
            calibration_error = torch.sqrt(calibration_error)

        return calibration_error

    def _compute_calibration_error(self, probas, targets, upper_bounds):
        probas = probas.view(-1)
        targets = targets.view(-1)

        bin_indices = torch.bucketize(probas, upper_bounds, right=True)
        if self.num_bins is None:
            self.num_bins = 0
        sums = torch.bincount(bin_indices, weights=probas, minlength=self.num_bins)
        sums = sums.float()
        counts = torch.bincount(bin_indices, minlength=self.num_bins)
        counts = counts + torch.finfo(sums.dtype).eps
        confidences = sums / counts
        accuracies = torch.bincount(bin_indices, weights=targets.float(), minlength=self.num_bins) / counts
        calibration_errors = accuracies - confidences

        if self.norm == 'l1':
            calibration_errors_normed = calibration_errors
        elif self.norm == 'l2':
            calibration_errors_normed = torch.square(calibration_errors)
        else:
            raise ValueError(f'Unknown norm, got {self.norm}')
        weighting = counts / probas.numel()
        weighted_calibration_errors = weighting * calibration_errors_normed

        error = torch.sum(torch.abs(weighted_calibration_errors))
        return error


class ExpectedCalibrationError(GeneralCalibrationError):
    def __init__(self, num_bins: int = 15):
        super().__init__(
            binning_scheme='even',
            max_prob=True,
            class_conditional=False,
            norm='l1',
            num_bins=num_bins,
            threshold=0
        )


class StaticCalibrationError(GeneralCalibrationError):
    def __init__(self, num_bins: int = 15, threshold: float = 0.01):
        super().__init__(
            binning_scheme='even',
            max_prob=False,
            class_conditional=True,
            norm='l1',
            num_bins=num_bins,
            threshold=threshold
        )


class AdaptiveCalibrationError(GeneralCalibrationError):
    def __init__(self, num_bins: int = 15, threshold: float = 0.01):
        super().__init__(
            binning_scheme='adaptive',
            max_prob=False,
            class_conditional=True,
            norm='l1',
            num_bins=num_bins,
            threshold=threshold
        )
