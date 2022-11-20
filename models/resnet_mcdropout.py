import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import generalization, calibration, ood
from utils import MetricLogger, SmoothedValue
from models.utils.bayesian_module import BayesianModule, ConsistentMCDropout2d


class DropoutResNet18(BayesianModule):
    def __init__(self, num_classes=10, k=None, dropout_rate=0.2):
        super(BayesianModule, self).__init__()
        self.in_planes = 64
        self.block = DropoutBasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.k = k
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], self.dropout_rate, stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], self.dropout_rate, stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], self.dropout_rate, stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], self.dropout_rate, stride=2)
        self.linear = nn.Linear(512*self.block.expansion, num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embeddings=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = (self.linear(out), out) if get_embeddings else self.linear(out)
        return out

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return self.forward(mc_input_BK)

    @torch.no_grad()
    def forward_logits(self, dataloader, device):
        #TODO: Should logits for querying be the mean of mc forward logits?
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = torch.mean(self.mc_forward(samples.to(device), k=self.k), dim=1)
            all_logits.append(logits)
        return torch.cat(all_logits)


class DropoutBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_dropout = ConsistentMCDropout2d(dropout_rate)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_dropout = ConsistentMCDropout2d(dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv1_dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
    dropout_logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        dropout_logits_id.append(model.mc_forward(inputs, model.k))
        targets_id.append(targets)

    # Transform to tensor
    dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Transform into probabilitys
    dropout_probas_id = dropout_logits_id.softmax(dim=-1)

    # Average of probas per sample
    mean_probas_id = torch.mean(dropout_probas_id, dim=1)
    mean_probas_id = ood.clamp_probas(mean_probas_id)

    # Confidence- and entropy-Scores of in domain set logits
    conf_id, _ = mean_probas_id.max(-1)
    entropy_id = ood.entropy_fn(mean_probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(torch.log(mean_probas_id), targets_id, (1,))[0].item()
    prec = generalization.avg_precision(mean_probas_id, targets_id)
    loss = criterion(torch.log(mean_probas_id), targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

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
        dropout_logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            dropout_logits_ood.append(model.mc_forward(inputs, model.k))
        dropout_logits_ood = torch.cat(dropout_logits_ood, dim=0).cpu()
        dropout_probas_ood = dropout_logits_ood.softmax(dim=-1)
        mean_probas_ood = torch.mean(dropout_probas_ood, dim=1)
        mean_probas_ood = ood.clamp_probas(mean_probas_ood)

        # Confidence- and entropy-Scores of out of domain logits
        conf_ood, _ = mean_probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(mean_probas_ood)

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Training: Epoch {epoch}"
    model.to(device)
    model.train()

    for X_batch, y_batch in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        loss = criterion(out, y_batch)
        batch_size = X_batch.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, = generalization.accuracy(out.softmax(dim=-1), y_batch, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats
    