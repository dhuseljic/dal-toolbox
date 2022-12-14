import torch

from ..metrics import generalization, ood, calibration
from ..utils import MetricLogger, SmoothedValue
from .utils.pseudo_labels import generate_mask, generate_pseudo_labels, freeze_bn, unfreeze_bn

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.num_classes = num_classes

        # Init layer does not have a kernel size of 7 since cifar has a smaller
        # size of 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512*self.block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        return probas

    @torch.inference_mode()
    def get_representation(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for samples, _ in dataloader:
            _, features = self(samples.to(device), return_features=True)
            all_features.append(features.cpu())
        features = torch.cat(all_features)
        return features

    @torch.inference_mode()
    def get_grad_embedding(self, dataloader, device):
        self.eval()
        self.to(device)
        feature_dim = 512

        embedding = []
        for inputs, _ in dataloader:
            embedding_batch = torch.empty([len(inputs), feature_dim * self.num_classes])
            logits, features = self(inputs.to(device), return_features=True)
            logits = logits.cpu()
            features = features.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            # TODO: optimize code
            # for each sample in a batch and for each class, compute the gradient wrt to weights
            for n in range(len(inputs)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding


def train_one_epoch(model, dataloaders, criterion, optimizer, device, n_epochs, unsup_warmup, 
                    lambda_u, p_cutoff, use_hard_labels, epoch=None, print_freq=200):
    model.train()
    model.to(device)
    criterion.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    for (x_lb, y_lb), (x_ulb, y_ulb) in zip(dataloaders['train_sup'], dataloaders['train_unsup']):
        x_lb, y_lb = x_lb.to(device), y_lb.to(device)
        x_ulb, y_ulb = x_ulb.to(device), y_ulb.to(device)

        # Outputs of labeled data
        logits_lb = model(x_lb)

        # Supervised loss
        sup_loss = criterion(logits_lb, y_lb)

        # Outputs of unlabeled data but without batch norm
        bn_backup = freeze_bn(model)
        logits_ulb = model(x_ulb)
        unfreeze_bn(model, bn_backup)

        # Generate pseudo labels in form of argmax of probas
        ps_lb = generate_pseudo_labels(logits_ulb, use_hard_labels)

        # Generate a mask in Form of a boolean tensor that is true if argmax of probas is > p_cutoff
        mask = generate_mask(logits_ulb, p_cutoff)

        # Unsupervised Loss for highly certain pseudo labels
        unsup_loss = (F.cross_entropy(logits_ulb, ps_lb, reduction='none') * mask).mean()

        # Calculate SSL Warm Up Factor
        unsup_warmup_ = torch.clip(torch.tensor(epoch / (unsup_warmup * n_epochs)),  min=0.0, max=1.0)

        # Loss thats used for backpropagation
        total_loss = sup_loss + unsup_warmup_ * lambda_u * unsup_loss

        # Update Model Weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        batch_size = x_lb.shape[0]
        acc1, = generalization.accuracy(logits_lb, y_lb, topk=(1,))
        metric_logger.update(
            sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(), 
            total_loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"],
            mask_ratio=mask.float().mean().item()
            )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats



@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Confidence- and entropy-Scores of in domain set logits
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    entropy_id = ood.entropy_fn(probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
    prec = generalization.avg_precision(probas_id, targets_id)
    loss = criterion(logits_id, targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

    metrics = {
        "acc1": acc1,
        "prec": prec,
        "loss": loss,
        "nll": nll,
        "tce": tce,
        "mce": mce
    }

    for name, dataloader_ood in dataloaders_ood.items():
        # Forward prop out of distribution
        logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_ood.append(model(inputs))
        logits_ood = torch.cat(logits_ood, dim=0).cpu()

        # Confidence- and entropy-Scores of out of domain logits
        probas_ood = logits_ood.softmax(-1)
        conf_ood, _ = probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(probas_ood)

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
