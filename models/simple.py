import torch
import torch.nn as nn
import numpy as np

from .spectral_norm import spectral_norm_fc
from sklearn.metrics import roc_auc_score
from torch.distributions import MultivariateNormal

from metrics.ood import ood_auroc


class ResNet(nn.Module):
    def __init__(self, n_classes, coeff, n_residuals, spectral_norm=True, n_power_iterations=1):
        super().__init__()
        self.first = nn.Conv2d(1, 32, kernel_size=5, padding=0, stride=2)
        self.residuals = nn.ModuleList([nn.Conv2d(32, 32, kernel_size=5, padding=2) for _ in range(n_residuals)])
        self.last = nn.Linear(288, n_classes)

        self.n_classes = n_classes
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.ELU()

        if spectral_norm:
            for residual in self.residuals:
                spectral_norm_fc(
                    residual,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations
                    # input_dim=[16, 6, 6],
                )
        self.means, self.covs, self.pis = None, None, None

    def forward(self, x):
        out = self.act(self.first(x))
        out = self.avg_pool(out)

        for residual in self.residuals:
            out = self.act(residual(out)) + out
        out = self.avg_pool(out).flatten(1)
        self.features = out

        out = self.last(out)
        return out

    def predict_gmm_log_prob(self, features):
        if self.means is None or self.covs is None or self.pis is None:
            raise ValueError
        log_probs = torch.stack([MultivariateNormal(mean, covariance_matrix=cov).log_prob(features)
                                 for mean, cov in zip(self.means, self.covs)], dim=1)
        log_probs = torch.logsumexp(torch.stack(self.pis).log() + log_probs, -1)
        return log_probs


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    model.to(device)

    # Train the epoch
    running_loss, running_corrects, n_samples = 0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        n_samples += batch_size
        running_loss += loss.item()*batch_size
        running_corrects += (outputs.argmax(-1) == targets).sum().item()
    train_stats = {'train_acc': running_corrects/n_samples, 'train_loss': running_loss/n_samples}

    # Fit for OOD
    features_accumulated, targets_accumulated = [], []
    for inputs, targets in dataloader:
        with torch.no_grad():
            outputs = model(inputs.to(device))
        features_accumulated.append(model.features.cpu())
        targets_accumulated.append(targets)
    features = torch.cat(features_accumulated)
    targets = torch.cat(targets_accumulated)

    means, covs, pis = [], [], []
    for k in range(model.n_classes):
        mask = (targets == k)
        means.append(features[mask].mean(0))
        covs.append(features[mask].T.cov())
        pis.append(mask.float().mean())
    model.means = means
    model.covs = covs
    model.pis = pis

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    test_stats = {}
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id, features_id = [], [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
        features_id.append(model.features)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()
    features_id = torch.cat(features_id, dim=0).cpu()

    # Update test stats
    test_stats.update({'test_loss': criterion(logits_id, targets_id).item()})
    test_stats.update({'test_acc': (logits_id.argmax(-1) == targets_id).float().mean().item()})

    # Forward prop out of distribution
    logits_ood, features_ood = [], []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model(inputs))
        features_ood.append(model.features)
    logits_ood = torch.cat(logits_ood, dim=0).cpu()
    features_ood = torch.cat(features_ood, dim=0).cpu()

    # Update test stats
    # GMM fitted on train features
    scores_id = -model.predict_gmm_log_prob(features_id)
    scores_ood = -model.predict_gmm_log_prob(features_ood)
    test_stats.update({'auroc': ood_auroc(scores_id, scores_ood)})

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood_auroc(1-conf_id, 1-conf_ood)})
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = - torch.sum(probas_id * probas_id.log(), dim=-1)
    entropy_ood = - torch.sum(probas_ood * probas_ood.log(), dim=-1)
    test_stats.update({'auroc_net_entropy': ood_auroc(entropy_id, entropy_ood)})

    return test_stats
