import torch
import torch.nn as nn


def calibration_error(confs: torch.Tensor, accs: torch.Tensor, n_samples: torch.Tensor, p: int = 2):
    probas_bin = n_samples/n_samples.nansum()
    ce = (torch.nansum(probas_bin * (confs-accs)**p))**(1/p)
    return ce


class TopLabelCalibrationError(nn.Module):
    """Computes the calibration plot for each class."""

    def __init__(self, n_bins=100, p=2):
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

    def __init__(self, n_bins=100, p=2):
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