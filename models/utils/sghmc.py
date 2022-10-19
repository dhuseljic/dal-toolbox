import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import generalization, ood


class HMCModel(nn.Module):

    def __init__(self, model, n_total_batches, n_snaphots=10, warm_up_batches=100):
        super().__init__()
        assert warm_up_batches < n_total_batches, f'Warm up ({warm_up_batches}) needs to be smaller than total batch size ({n_total_batches}).'
        self.model = model

        self.n_snapshots = n_snaphots
        self.n_total_batches = n_total_batches
        self.warm_up_batches = warm_up_batches

        self.snapshot_interval = (n_total_batches - warm_up_batches) // n_snaphots

        self.update_step = 0

        self.snapshots = []

    def forward(self, x):
        return self.model.forward(x)

    def save_snapshot_step(self):
        # Save snapshots after it has seen batch batches for training
        self.update_step += 1
        if ((self.update_step-self.warm_up_batches+1) % self.snapshot_interval) == 0 and self.update_step >= self.warm_up_batches:
            weights = copy.deepcopy(self.model.state_dict())
            self.snapshots.append(weights)

    @torch.no_grad()
    def forward_snapshots(self, x):
        logits = []
        for weights in self.snapshots:
            self.model.load_state_dict(weights)
            out = self.forward(x)
            logits.append(out)
        return torch.stack(logits, dim=0)


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
        model.save_snapshot_step()

        batch_size = inputs.shape[0]
        acc1, = generalization.accuracy(outputs, targets, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    if len(model.snapshots) <= 1:
        return {}  # vanilla.evaluate(model, dataloader_id, dataloader_ood, criterion, device)
    model.eval()
    model.to(device)
    test_stats = {}

    logits_id, targets_id = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model.forward_snapshots(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=1).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Ensemble accuracy
    probas = logits_id.softmax(-1)
    acc1, = generalization.accuracy(probas.mean(0), targets_id, topk=(1,))
    loss = F.nll_loss(probas.mean(0).log(), targets_id)
    test_stats.update({'acc1': acc1.item(), 'loss': loss.item()})
    

    # Snapshot accuracies
    for i_snapshot, logits in enumerate(logits_id):
        loss = criterion(logits, targets_id)
        acc1, = generalization.accuracy(logits, targets_id, topk=(1,))
        test_stats.update({
            f'snapshot{i_snapshot}_loss': loss.item(),
            f'snapshot{i_snapshot}_acc1': acc1.item(),
        })

    logits_ood, targets_ood = [], []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model.forward_snapshots(inputs))
        targets_ood.append(targets)
    logits_ood = torch.cat(logits_ood, dim=1).cpu()
    targets_ood = torch.cat(targets_ood, dim=0).cpu()

    # Compute
    probas_id = logits_id.softmax(-1).mean(0)
    probas_ood = logits_ood.softmax(-1).mean(0)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    test_stats.update({'auroc': ood.ood_auroc(entropy_id, entropy_ood)})

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats


class SGHMC(torch.optim.Optimizer):
    def __init__(self, params, n_samples, lr=0.001, prior_precision=1, M=1.0, C=10, B_estim=0.0, resample_each=100):
        if lr < 0.0:
            raise ValueError(f"Invalid setp size (learning rate): {lr} - should be >=0.0")
        if prior_precision < 0.0:
            raise ValueError(f"Invalid prior_precision: {prior_precision} - should be >=0.0")
        defaults = dict(
            n_samples=n_samples,
            lr=lr,
            prior_precision=prior_precision,
            M=M,
            C=C,
            B_estim=B_estim,
            resample_each=resample_each
        )
        super().__init__(params, defaults)
        # dicitionary with momentums of all parameters
        self.r = dict()
        self.t = 0

    def step(self):
        self.t += 1
        # iterate groups
        for group in self.param_groups:
            n_samples = group['n_samples']
            lr = group['lr']
            prior_precision = group['prior_precision']
            M = group['M']
            C = group['C']
            B_estim = group['B_estim']
            resample_each = group['resample_each']

            # iterate parameter in current group
            for p in group['params']:
                if not p.requires_grad:
                    continue
                ########################################################################
                ## compute noisy estimate of the energy function gradient             ##
                ##  where: p(theta|data) \propto exp(-U(theta))                       ##
                ########################################################################
                # gradient with respect to log likelihood, where
                #   p.grad.data is assumed to show the gradient of the "mean" of single
                #   instance likelihood gradients
                ll_grad = n_samples*p.grad.data
                # exact gradient with respect to log prior
                lprior_grad = prior_precision*p.data
                # estimate of the posterior grad d(p(theta|data))/d(theta)
                grad = lprior_grad + ll_grad

                # check, if current parameters momentum is already available in dicitionary
                if p in self.r.keys():
                    if self.t % resample_each == 0:
                        self.r[p] = math.sqrt(M)*torch.randn(*p.data.shape, device=p.device)
                else:
                    self.r[p] = math.sqrt(M)*torch.randn(*p.data.shape, device=p.device)

                p.data.add_(lr/M*self.r[p])
                self.r[p] = self.r[p] - lr*grad - lr*C/M*self.r[p] + \
                    math.sqrt(2*(C-B_estim)*lr)*torch.randn(*p.data.shape, device=p.device)
