import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal

from utils import MetricLogger, SmoothedValue
from metrics import ood, generalization


class DDUWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.means = None
        self.covs = None
        self.pis = None

    def forward(self, x, return_features=False):
        return self.model(x, return_features=return_features)

    def predict_gmm_log_prob(self, features):
        if self.means is None or self.covs is None or self.pis is None:
            raise ValueError
        log_probs = torch.empty(len(features), len(self.pis))
        for i, (mean, cov) in enumerate(zip(self.means, self.covs)):
            dist = MultivariateNormal(mean, cov)
            log_prob = dist.log_prob(features)
            log_probs[:, i] = log_prob
        # log_probs1 = torch.stack([MultivariateNormal(mean, covariance_matrix=cov).log_prob(features)
        #                          for mean, cov in zip(self.means, self.covs)], dim=1)
        log_probs = torch.logsumexp(torch.stack(self.pis).log() + log_probs, -1)
        return log_probs


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    # Train the epoch
    for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        acc1, acc5 = generalization.accuracy(outputs, targets, topk=(1, 5))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    # Fit for OOD
    features_accumulated, targets_accumulated = [], []
    for inputs, targets in dataloader:
        with torch.no_grad():
            outputs, features = model(inputs.to(device), return_features=True)
        features_accumulated.append(features.cpu())
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
        outputs, features = model(inputs, return_features=True)
        logits_id.append(outputs)
        targets_id.append(targets)
        features_id.append(features)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()
    features_id = torch.cat(features_id, dim=0).cpu()

    # Update test stats
    loss = criterion(logits_id, targets_id)
    acc1, acc5 = generalization.accuracy(logits_id, targets_id, (1, 5))
    test_stats.update({'test_loss': loss.item()})
    test_stats.update({'test_acc1': acc1.item()})
    test_stats.update({'test_acc5': acc5.item()})

    # Forward prop out of distribution
    logits_ood, features_ood = [], []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, features = model(inputs, return_features=True)
        logits_ood.append(outputs)
        features_ood.append(features)
    logits_ood = torch.cat(logits_ood, dim=0).cpu()
    features_ood = torch.cat(features_ood, dim=0).cpu()

    # Update test stats
    # GMM fitted on train features
    scores_id = -model.predict_gmm_log_prob(features_id)
    scores_ood = -model.predict_gmm_log_prob(features_ood)
    test_stats.update({'auroc': ood.ood_auroc(scores_id, scores_ood)})

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood.ood_auroc(1-conf_id, 1-conf_ood)})
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    test_stats.update({'auroc_net_entropy': ood.ood_auroc(entropy_id, entropy_ood)})

    return test_stats