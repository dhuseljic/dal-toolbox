import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics


class BrierScoreDecomposition(nn.Module):
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


def log_sub_exp(x, y):
    larger = torch.max(x, y)
    smaller = torch.min(x, y)
    zero = torch.zeros_like(larger)
    result = larger + log1mexp(torch.max(larger - smaller, zero))
    return result


def log1mexp(x,):
    x = torch.abs(x)
    return torch.where(x < math.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


class BrierScore(nn.Module):
    def forward(self, probas, labels):
        n_samples, n_classes = probas.shape
        assert len(labels) == n_samples, "Probas and Labels must be of the same size"
        labels_onehot = F.one_hot(labels, num_classes=n_classes)
        score = torch.sum(F.mse_loss(probas, labels_onehot, reduction='none'), -1)
        score = torch.mean(score)

        # Note: Tensorflow ignores the addition by one. To be consistent with the decomposition, we also ignore it.
        # probas_label = torch.sum(probas * labels_onehot, -1)
        # score = torch.sum(probas**2, dim=-1) - 2 * probas_label +1
        # score = torch.mean(score, -1)
        return score


class EnsembleCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits.shape = (EnsembleMembers, Samples, Classes)
        labels.shape = (Samples)
        """
        # Remember ensemble_size for nll calculation
        ensemble_size, n_samples, n_classes = logits.shape

        # Reshape to fit CrossEntropy
        # logits.shape = (EnsembleMembers*Samples, Classes)
        # labels.shape = (EnsembleMembers*Sampels)
        labels = torch.broadcast_to(labels.unsqueeze(0), logits.shape[:-1])
        labels = labels.reshape(ensemble_size*n_samples)
        logits = logits.reshape(ensemble_size*n_samples, n_classes)

        # Non Reduction Cross Entropy
        ce = self.cross_entropy(logits, labels).reshape(-1, 1)

        # Reduce LogSumExp + log of Ensemble Size
        nll = -torch.logsumexp(-ce, dim=1) + math.log(ensemble_size)

        # Return Average
        return torch.mean(nll)


class GibsCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits.shape = (EnsembleMembers, Samples, Classes)
        labels.shape = (Samples)
        """
        # Remember ensemble_size for nll calculation
        ensemble_size, n_samples, n_classes = logits.shape

        # Reshape to fit CrossEntropy
        labels = torch.broadcast_to(labels.unsqueeze(0), logits.shape[:-1])
        labels = labels.reshape(ensemble_size*n_samples)
        logits = logits.reshape(ensemble_size*n_samples, n_classes)

        # Non Reduction Cross Entropy
        nll = self.cross_entropy(logits, labels).reshape(-1, 1)

        # Return Average
        return torch.mean(nll)


def calibration_error(confs: torch.Tensor, accs: torch.Tensor, n_samples: torch.Tensor, p: int = 2):
    probas_bin = n_samples/n_samples.nansum()
    ce = (torch.nansum(probas_bin * torch.abs(confs-accs)**p))**(1/p)
    return ce


class TopLabelCalibrationError(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=15, p=1):
        super().__init__()
        self.n_bins = n_bins
        self.p = p

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.n_bins+1)

        confs = torch.Tensor(self.n_bins)
        accs = torch.Tensor(self.n_bins)
        n_samples = torch.Tensor(self.n_bins)

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
        return calibration_error(confs, accs, n_samples, self.p)


class MarginalCalibrationError(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=15, p=2):
        super().__init__()
        self.n_bins = n_bins
        self.p = p

    def forward(self, probas: torch.Tensor, labels: torch.Tensor):
        bins = torch.linspace(0, 1, self.n_bins+1)
        _, n_classes = probas.shape

        # Save calibration plots in results
        self.results = []
        for i_cls in range(n_classes):
            label = (labels == i_cls).long()
            proba = probas[:, i_cls]

            confs = torch.Tensor(self.n_bins)
            accs = torch.Tensor(self.n_bins)
            n_samples = torch.Tensor(self.n_bins)
            for i_bin, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                in_bin = (bin_start < proba) & (proba < bin_end)
                n_samples[i_bin] = in_bin.sum()

                if in_bin.sum() == 0:
                    confs[i_bin] = float('nan')
                    accs[i_bin] = float('nan')
                    continue

                bin_conf = proba[in_bin].mean()
                bin_acc = (label[in_bin] == 1).float().mean()

                confs[i_bin] = bin_conf
                accs[i_bin] = bin_acc
            self.results.append({'confs': confs, 'accs': accs, 'n_samples': n_samples, 'class': i_cls})

        sq_ces = [calibration_error(d['confs'], d['accs'], d['n_samples'], self.p)**self.p for d in self.results]
        mce = torch.Tensor(sq_ces).mean()**(1/self.p)
        return mce
