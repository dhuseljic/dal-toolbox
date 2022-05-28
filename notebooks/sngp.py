# %%
# fmt: off
import sys

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


hparams = {
    'coef1f': .9,
    'n_residuals': 6,
    'num_inducing': 1024,
    'kernel_scale': 10,
    'auto_scale_kernel': False,
    'epochs': 100,
    'momentum': .999,
    'ridge_penalty': 1e-6,
    'weight_decay': 1e-2,
}
torch.manual_seed(0)
model = SNGP(
    model=Net(coeff=hparams['coeff'], n_residual_layers=hparams['n_residuals']),
    in_features=128,
    num_classes=2,
    num_inducing=hparams['num_inducing'],
    kernel_scale=hparams['kernel_scale'],
    momentum=hparams['momentum'],
    auto_scale_kernel=hparams['auto_scale_kernel'],
    ridge_penalty=hparams['ridge_penalty']
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=hparams['weight_decay'])
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=hparams['weight_decay'])
criterion = nn.CrossEntropyLoss()
for i in range(hparams['epochs']):
    train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i)
# %%
"""Plot contours."""

model.eval()
domain = 5
xx, yy = torch.meshgrid(torch.linspace(-domain, domain, 51), torch.linspace(-domain, domain, 51))
zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

with torch.no_grad():
    logits, cov = model(zz, return_cov=True, update_precision=False)

# torch.distributions.MultivariateNormal(logits[:, 1], )
# mean field approx
logits = mean_field_logits(logits, cov)
probas = logits.softmax(-1)
zz = probas[:, 1].view(xx.shape)

print(hparams)
# plt.title(f"Ep {}")
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.colorbar()
plt.show()

# %%
