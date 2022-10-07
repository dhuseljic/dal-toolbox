# %%
# fmt: off
import sys
import math

sys.path.append('../')

import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import datasets
from models import sngp
from backbones.spectral_norm import SpectralLinear
# fmt: on

# %%
X, y = datasets.make_moons(100, noise=.1)
# X, y = datasets.make_circles(500, noise=.02)
X = (X - X.mean(0)) / X.std(0)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

train_ds = torch.utils.data.TensorDataset(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%


class Net(nn.Module):
    def __init__(self,
                 num_classes,
                 n_residual_layers=6,
                 feature_dim=128,
                 spectral_norm=True,
                 norm_bound=1,
                 n_power_iterations=1):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.residuals = nn.ModuleList([
            SpectralLinear(
                feature_dim,
                feature_dim,
                spectral_norm=spectral_norm,
                norm_bound=norm_bound,
                n_power_iterations=n_power_iterations
            ) for _ in range(n_residual_layers)])
        self.last = nn.Linear(feature_dim, num_classes)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0)

    def forward(self, x, return_features=False):
        # : Added activation to first layer
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.dropout(self.act(residual(x))) + x
        features = x
        x = self.last(x)
        if return_features:
            return x, features
        return x

# %%


@torch.no_grad()
def plot_contour(model, X, y, ax=None):
    if ax:
        plt.sca(ax)
    domain = 5
    xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

    logits, cov = model(zz, return_cov=True, update_precision=False)
    logits = sngp.mean_field_logits(logits, cov)

    probas = logits.softmax(-1)
    zz = probas[:, 1].view(xx.shape)

    print(spectral_hparams, gp_hparams)
    # plt.title(f"Ep {}")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
    plt.colorbar()
# %%


spectral_hparams = dict(
    norm_bound=.9,
    n_residual_layers=1,
    spectral_norm=True,
)
gp_hparams = dict(
    num_inducing=4*1024,
    kernel_scale=2,             # works like bandwidth
    normalize_input=False,      # important to disable
    scale_random_features=True,  # important to enable
    # Not that important, for inference
    cov_momentum=-1,
    ridge_penalty=1,
    mean_field_factor=math.pi/8,
)
epochs = 200
weight_decay = 1e-2

torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
model = sngp.SNGP(model=Net(num_classes=2, **spectral_hparams), in_features=128, num_classes=2, **gp_hparams)
nn.init.normal_(model.random_features.random_feature_linear.weight)
nn.init.xavier_normal_(model.beta.weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
history = []
for i in range(epochs):
    train_stats = sngp.train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i)
    history.append(train_stats)

model.eval()
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot([d['train_loss'] for d in history])
plot_contour(model, X, y, ax=plt.subplot(122))
plt.show()


# %%
"""Implement reweighting"""
torch.manual_seed(1)
domain = 5
draws = 10000
n_samples_new = 10

plt.figure(figsize=(15, 5))
plt.subplot(121)
xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

logits = model.forward_sample(zz, n_draws=draws, resample=True)
probas = logits.softmax(-1)
probas = probas.mean(0)[:, 1].view(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, s=10)

# p = logits.softmax(-1)[74]
# probas = p[:,1].view(xx.shape)
# c = plt.contourf(xx, yy, probas, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
# plt.colorbar(c)

c = plt.contourf(xx, yy, probas, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.colorbar(c)
plt.contour(xx, yy, probas, alpha=.3, zorder=-1, levels=[0.5], colors='black')


plt.subplot(122)

# Create new Cluster
noise = .2
X_new = torch.cat([
    torch.randn((n_samples_new, 2))*noise + 3,
    torch.randn((n_samples_new, 2))*noise - 3,
    torch.randn((n_samples_new, 2))*noise - torch.Tensor([3, -3]),
    torch.randn((n_samples_new, 2))*noise + torch.Tensor([3, -3]),
    torch.randn((n_samples_new, 2))*noise + torch.Tensor([0, -3]),
    torch.randn((n_samples_new, 2))*noise + torch.Tensor([0, 3]),
    torch.randn((n_samples_new, 2))*noise + torch.Tensor([-3, 0]),
    torch.randn((n_samples_new, 2))*noise + torch.Tensor([3, 0]),
    # torch.randn((n_samples_new, 2))*2,
])
y_new = torch.cat([
    torch.ones(n_samples_new).long()*1,
    torch.ones(n_samples_new).long()*1,
    torch.ones(n_samples_new).long()*0,
    torch.ones(n_samples_new).long()*0,
    torch.ones(n_samples_new).long()*1,
    torch.ones(n_samples_new).long()*0,
    torch.ones(n_samples_new).long()*1,
    torch.ones(n_samples_new).long()*0,
    # torch.ones(n_samples_new).long()*0,
])


plt.scatter(X_new[:, 0], X_new[:, 1], s=10, c=y_new)

# Reweighting
lmb = .1
log_probas_sampled = model.forward_sample(X_new, n_draws=draws).log_softmax(-1)
log_weights = torch.log(torch.ones(draws) / draws)  # uniform prior
log_weights += lmb*log_probas_sampled[:, torch.arange(len(X_new)), y_new].sum(dim=1)
# normalize log probs numerically stable
weights = torch.exp(log_weights - log_weights.max())
weights /= weights.sum()

# for l, w in zip(logits, weights):
#     p = l.softmax(-1)
#     probas = p[:,1].view(xx.shape)
#     plt.contour(xx, yy, probas, alpha=.6, zorder=-1, levels=[0.5], colors='red', linewidths=w)

# Reweighted probas
probas_reweighted = torch.einsum('e,enk->nk', weights, logits.softmax(-1))
# plt.contourf(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.6, zorder=-1, levels=[0.5], colors='purple')
plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
c = plt.contourf(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.contour(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.3, zorder=-1, levels=[0.5], colors='black')
plt.colorbar(c)


# %%
plt.bar(range(len(weights)), weights)

# %%

dist = torch.distributions.MultivariateNormal(loc=model.beta.weight.data, precision_matrix=model.precision_matrix)
log_weights = dist.log_prob(model.sampled_betas).sum(-1)
plt.hist(log_weights.numpy())
# %%
"""SNGP Random Sequence."""

# INIT model
spectral_hparams = dict(
    norm_bound=.9,
    n_residual_layers=1,
    spectral_norm=True,
)
gp_hparams = dict(
    num_inducing=4*1024,
    kernel_scale=.1,             # works like bandwidth
    normalize_input=False,      # important to disable
    scale_random_features=True,  # important to enable
    # Not that important, for inference
    cov_momentum=-1,
    ridge_penalty=1,
    mean_field_factor=math.pi/8,
)
epochs = 200
weight_decay = 1e-2

torch.manual_seed(0)
model = sngp.SNGP(model=Net(num_classes=2, **spectral_hparams), in_features=128, num_classes=2, **gp_hparams)
nn.init.normal_(model.random_features.random_feature_linear.weight)
nn.init.xavier_normal_(model.beta.weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

domain = 2
draws = 10000

# Subset training
n_samples_train = 20
indices = range(n_samples_train)
train_ds_subset = torch.utils.data.Subset(train_ds, indices=indices)
train_loader = torch.utils.data.DataLoader(train_ds_subset, batch_size=128, shuffle=True)

# Subset Bayesian updating
n_samples_updating = 50
indices_updating= range(n_samples_train, n_samples_train+n_samples_updating)
# train_ds_updating = torch.utils.data.Subset(train_ds, indices=indices_updating)

for i_epoch in range(epochs):
    train_stats = sngp.train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i_epoch)

plt.figure(figsize=(15, 8))
plt.subplot(121)
xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

logits = model.forward_sample(zz, n_draws=draws, resample=True)
probas = logits.softmax(-1)
probas = probas.mean(0)[:, 1].view(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.scatter(X[indices, 0], X[indices, 1], c=y[indices], s=20)

c = plt.contourf(xx, yy, probas, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.colorbar(c)
plt.contour(xx, yy, probas, alpha=.3, zorder=-1, levels=[0.5], colors='black')

plt.subplot(122)

# Reweighting
lmb = .5
log_probas_sampled = model.forward_sample(X[indices_updating], n_draws=draws).log_softmax(-1)
log_weights = torch.log(torch.ones(draws) / draws)  # uniform prior
log_weights += lmb*log_probas_sampled[:, torch.arange(len(X[indices_updating])), y[indices_updating]].sum(dim=1)
# normalize log probs numerically stable
weights = torch.exp(log_weights - log_weights.max())
weights /= weights.sum()
probas_reweighted = torch.einsum('e,enk->nk', weights, logits.softmax(-1))

# plt.contourf(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.6, zorder=-1, levels=[0.5], colors='purple')
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.scatter(X[indices_updating, 0], X[indices_updating, 1], c=y[indices_updating], s=20)
c = plt.contourf(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.contour(xx, yy, probas_reweighted[:, 1].view(xx.shape), alpha=.3, zorder=-1, levels=[0.5], colors='black')
plt.colorbar(c)


# %%
