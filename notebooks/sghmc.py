# %%
# fmt: off
import sys
import math

from models.utils import sghmc

sys.path.append('../')

import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets

from models import sngp
# fmt: on
# %%

X, y = datasets.make_moons(500, noise=.1)
# X, y = datasets.make_circles(500, noise=.05)
# X, y = datasets.make_blobs([100, 200, 300, 400])
y %= 2
X = (X - X.mean(0)) / X.std(0)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

train_ds = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
# %%


class Net(nn.Module):
    def __init__(self,  feature_dim=128, kernel_scale=1, num_inducing=1024, scale_features=True):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.hidden1 = nn.Linear(feature_dim, feature_dim)
        self.hidden2 = nn.Linear(feature_dim, feature_dim)
        self.random_features = sngp.RandomFourierFeatures(
            feature_dim,
            num_inducing,
            kernel_scale=kernel_scale,
            scale_features=scale_features
        )
        self.last = nn.Linear(num_inducing, 2)

        self.act = nn.ReLU()

    def forward(self, x):
        # : Added activation to first layer
        x = self.act(self.first(x))
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.random_features(x)
        x = self.last(x)
        return x


# %%

hparams = {
    'n_epochs': 100
}
torch.manual_seed(0)
# DNN
hparams_net = dict(
    feature_dim=128,
    kernel_scale=.1,
    num_inducing=1024,
    scale_features=True,
)
net = Net(**hparams_net)
# sghmc
hparams_model = dict(
    n_total_batches=hparams['n_epochs']*len(train_loader),
    n_snaphots=20,
    warm_up_batches=400,
)
model = sghmc.HMCModel(model=net, **hparams_model)
# Optimizer
epsilon = 0.1/math.sqrt(len(train_ds))
C = .1 / epsilon
B_estim = .8*C
resample_each = int(1e10)
hparams_sghmc = dict(
    n_samples=len(train_ds),
    epsilon=epsilon,
    C=C,
    B_estim=B_estim,
    resample_each=resample_each,
    prior_precision=1,
)
optimizer = sghmc.SGHMC(model.parameters(), **hparams_sghmc)
criterion = nn.CrossEntropyLoss()

for i in range(hparams['n_epochs']):
    sghmc.train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i)

# %%

plot_snapshots = True
print(hparams_model, hparams_sghmc)
model.eval()
xx, yy = torch.meshgrid(torch.linspace(-5, 5, 51), torch.linspace(-5, 5, 51))
zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

logits_snapshots = []
with torch.no_grad():
    logits = model.forward_snapshots(zz)
    for weights in model.snapshots:
        model.model.load_state_dict(weights)
        logits_snapshots.append(model(zz))

zz = logits.softmax(-1).mean(0)[:, 1]
zz = zz.view(xx.shape)

plt.title(f"Ep {i}: {hparams}")
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.colorbar()
if plot_snapshots:
    for logits in logits_snapshots:
        zz = logits.softmax(-1)[:, 1]
        zz = zz.view(xx.shape)
        plt.contour(xx, yy, zz, alpha=.3, zorder=-1, levels=[.5], colors='black')
plt.show()

# %%
