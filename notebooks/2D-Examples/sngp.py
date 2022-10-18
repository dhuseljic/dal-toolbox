# %%
# fmt: off
import sys
import math

sys.path.append('../../')

import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import datasets
from models.utils.spectral_norm import SpectralLinear
from models.resnet_sngp import RandomFeatureGaussianProcess, train_one_epoch
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


class SNGPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 spectral_norm_params: dict = {},
                 gp_params: dict = {},
                 n_residual_layers: int = 6,
                 feature_dim: int = 128,
                 ):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.residuals = nn.ModuleList([
            SpectralLinear(
                feature_dim,
                feature_dim,
                **spectral_norm_params
            ) for _ in range(n_residual_layers)])
        self.last = RandomFeatureGaussianProcess(
            in_features=feature_dim,
            out_features=num_classes,
            **gp_params,
        )
        self.act = nn.ReLU()

    def reset_precision_matrix(self):
        self.last.reset_precision_matrix()

    def forward(self, x, mean_field=False, return_cov=False):
        # : Added activation to first layer
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x) + x)

        if mean_field:
            out = self.last.forward_mean_field(x)
        else:
            out = self.last(x, return_cov=return_cov)

        return out

# %%


@torch.no_grad()
def plot_contour(model, X, y, ax=None):
    model.cpu()
    if ax:
        plt.sca(ax)
    domain = 5
    xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

    logits = model(zz, mean_field=True)
    probas = logits.softmax(-1)
    zz = probas[:, 1].view(xx.shape)

    # plt.title(f"Ep {}")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
    plt.colorbar()


# %%
spectral_norm_params = dict(
    spectral_norm=True,
    norm_bound=.9,
    n_power_iterations=1
)
gp_params = dict(
    num_inducing=1024,
    kernel_scale=1,
    normalize_input=False,
    scale_random_features=False,
    # mean_field_factor=np.pi/8,
    # cov_momentum=-1,
    # ridge_penalty=1
)
optimizer_params = dict(
    lr=1e-3,
    weight_decay=1e-2,
    momentum=.9,
)
epochs=200

torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

model = SNGPNet(num_classes=2, spectral_norm_params=spectral_norm_params, gp_params=gp_params)
model.last.random_features.reset_parameters(std_init=1)
optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
criterion = nn.CrossEntropyLoss()
history = []
for i in range(epochs):
    train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device='cuda', epoch=i)
    history.append(train_stats)

model.eval()
plt.figure(figsize=(15, 5))
plt.subplot(122)
plt.plot([d['train_loss'] for d in history])
plot_contour(model, X, y, ax=plt.subplot(121))
plt.show()
# %%
