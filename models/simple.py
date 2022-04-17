import torch
import torch.nn as nn

from .spectral_norm import spectral_norm_fc
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score


class ResNet(nn.Module):
    def __init__(self, n_classes, coeff, n_residuals, spectral_norm=True, n_power_iterations=1):
        super().__init__()
        self.first = nn.Conv2d(1, 16, kernel_size=5, padding=0, stride=2)
        self.residuals = nn.ModuleList([nn.Conv2d(16, 16, kernel_size=5, padding=2) for _ in range(n_residuals)])
        self.last = nn.Linear(144, n_classes)
        self.n_classes = n_classes

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.ELU()

        if spectral_norm:
            spectral_norm_fc(self.first, coeff=coeff, n_power_iterations=n_power_iterations)
            for residual in self.residuals:
                spectral_norm_fc(
                    residual,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations
                    # input_dim=[16, 6, 6],
                )

    def forward(self, x):
        out = self.act(self.first(x))
        out = self.avg_pool(out)

        for residual in self.residuals:
            out = self.act(residual(out)) + out
        out = self.avg_pool(out).flatten(1)
        self.features = out

        out = self.last(out)
        return out


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    model.to(device)

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
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': auroc_fn(1-conf_id, 1-conf_ood)})
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = - torch.sum(probas_id * probas_id.log(), dim=-1)
    entropy_ood = - torch.sum(probas_ood * probas_ood.log(), dim=-1)
    test_stats.update({'auroc_net_entropy': auroc_fn(entropy_id, entropy_ood)})
    # gmm auroc
    gmm = GaussianMixture(n_components=model.n_classes)
    gmm.fit(features_id)
    log_likelihood_id = -torch.from_numpy(gmm.score_samples(features_id))
    log_likelihood_ood = -torch.from_numpy(gmm.score_samples(features_ood))
    test_stats.update({'auroc_gmm': auroc_fn(log_likelihood_id, log_likelihood_ood)})

    return test_stats


def auroc_fn(score_id: torch.Tensor, score_ood: torch.Tensor):
    y_true = torch.cat((torch.zeros(len(score_id)), torch.ones(len(score_ood))))
    y_score = torch.cat((score_id, score_ood))
    return roc_auc_score(y_true, y_score)
