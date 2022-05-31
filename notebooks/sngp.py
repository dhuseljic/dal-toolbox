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
from models.sngp import SNGP, mean_field_logits, train_one_epoch
from backbones.spectral_norm import spectral_norm_fc
# fmt: on

# %%
X, y = datasets.make_moons(500, noise=.1)
# X, y = datasets.make_circles(500, noise=.02)
X = (X - X.mean(0)) / X.std(0)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

train_ds = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%


class Net(nn.Module):
    def __init__(self, n_residual_layers=4, feature_dim=128, spectral_norm=True, coeff=1, n_power_iterations=1):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.residuals = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(n_residual_layers)])
        self.last = nn.Linear(feature_dim, 2)

        # Add spectral norm
        if spectral_norm:
            for residual in self.residuals:
                spectral_norm_fc(
                    residual,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations
                )

        self.act = nn.ELU()

    def forward(self, x, return_features=False):
        # : Added activation to first layer
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x)) + x
        features = x
        x = self.last(x)
        if return_features:
            return x, features
        return x

# %%


def plot_contour(model, X, y, ax=None):
    if ax:
        plt.sca(ax)
    model.eval()
    domain = 5
    xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

    with torch.no_grad():
        logits, cov = model(zz, return_cov=True, update_precision=False)

    logits = model.scale_logits(logits, cov)
    probas = logits.softmax(-1)
    zz = probas[:, 1].view(xx.shape)

    print(spectral_hparams, gp_hparams)
    # plt.title(f"Ep {}")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
    plt.colorbar()
# %%

spectral_hparams = dict(
    coeff=.9,
    n_residual_layers=6,
)
gp_hparams = dict(
    num_inducing=1024,
    kernel_scale=.1,             # works like bandwidth
    normalize_input=False,      # important to disable
    scale_random_features=True, # important to enable
    # Not that important, for inference
    cov_momentum=-1,
    ridge_penalty=1e-6,
    mean_field_factor=math.pi/8,
)
epochs = 100
weight_decay = 0

torch.manual_seed(0)
model = SNGP(model=Net(**spectral_hparams), in_features=128, num_classes=2, **gp_hparams)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=weight_decay, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
history = []
for i in range(epochs):
    train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i)
    history.append(train_stats)

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot([d['train_loss'] for d in history])
plot_contour(model, X, y, ax=plt.subplot(122))
plt.show()

# %%
