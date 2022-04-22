import math
import copy

import torch
import torch.nn as nn

from utils import MetricLogger
from metrics import ood, generalization


class SNGPWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 in_features: int,
                 num_inducing: int,
                 num_classes: int,
                 kernel_scale: float = None,
                 ):
        super().__init__()
        self.model = model

        self.layer_norm = nn.LayerNorm(in_features)

        # Define scale, weight and bias according to Eq. 7
        self.scale = math.sqrt(2./float(num_inducing))
        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        nn.init.normal_(self.random_feature_linear.weight)
        # Scale the random features according to SNGP, bigger sigma equals bigger kernel
        if kernel_scale is None:
            kernel_scale = math.sqrt(in_features / 2)
        self.random_feature_linear.weight.data /= kernel_scale
        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

        # Define learnable weight beta according to Eq 8.
        self.beta = nn.Linear(num_inducing, num_classes, bias=False)
        nn.init.normal_(self.beta.weight)

        self.init_precision_matrix = torch.eye(num_inducing)
        self.reset_cov()

    def reset_cov(self):
        self.precision_matrix = nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)

    def compute_covariance(self, feature):
        delta = 1e-3  # TODO: multiply??
        covariance_matrix = torch.cholesky_inverse(self.precision_matrix)
        covariance_matrix = feature @ covariance_matrix @ feature.T
        return covariance_matrix

    def forward(self, x, update_cov=False, return_cov=False):
        _, features = self.model(x, return_features=True)

        # TODO: normalize layer? with layer norm
        features = self.layer_norm(features)
        # features = features * (1./math.sqrt(2))

        # Eq. 7
        phi = self.scale * torch.cos(self.random_feature_linear(features))

        # Eq. 8
        g = self.beta(phi)

        # Compute cov
        if update_cov:
            momentum = .999  # TODO: momentum mini batch
            pre_mat_minibatch = torch.matmul(phi.T, phi) / len(x)
            pre_mat_new = momentum * self.precision_matrix + (1. - momentum) * pre_mat_minibatch
            self.precision_matrix = nn.Parameter(copy.deepcopy(pre_mat_new), requires_grad=False)

        output = g
        if return_cov:
            cov = self.compute_covariance(phi)
            output = [output, cov]

        return output


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
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
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    test_stats = {}
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Update test stats
    loss = criterion(logits_id, targets_id)
    acc1, acc5 = generalization.accuracy(logits_id, targets_id, (1, 5))
    test_stats.update({'test_loss': loss.item()})
    test_stats.update({'test_acc1': acc1.item()})
    test_stats.update({'test_acc5': acc5.item()})

    # Forward prop out of distribution
    logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model(inputs))
    logits_ood = torch.cat(logits_ood, dim=0).cpu()

    # Update test stats
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    test_stats.update({'auroc': ood.ood_auroc(entropy_id, entropy_ood)})
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood.ood_auroc(1-conf_id, 1-conf_ood)})

    return test_stats
