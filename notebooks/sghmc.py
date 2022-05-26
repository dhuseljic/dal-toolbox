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

from models import sghmc
from utils import MetricLogger
from metrics import generalization
# fmt: on
# %%

X, y = datasets.make_moons(500, noise=.1)
X = (X - X.mean(0)) / X.std(0)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

train_ds = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
# %%


class Net(nn.Module):
    def __init__(self,  feature_dim=128):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.hidden1 = nn.Linear(feature_dim, feature_dim)
        self.hidden2 = nn.Linear(feature_dim, feature_dim)
        self.last = nn.Linear(feature_dim, 2)

        self.act = nn.ELU()

    def forward(self, x, return_features=False):
        # : Added activation to first layer
        x = self.act(self.first(x))
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.last(x)
        return x


# %%

hparams = {
    'n_epochs': 100
}
torch.manual_seed(0)
model = sghmc.HMCModel(
    model=Net(feature_dim=128),
    n_total_batches=hparams['n_epochs']*len(train_loader),
    n_snaphots=20,
    warm_up_batches=0,
)
epsilon = 0.1/math.sqrt(len(train_ds))
C = .1 / epsilon
B_estim = .9*C
resample_each = int(1e10)
optimizer = sghmc.SGHMC(
    model.parameters(),
    len(train_ds),
    epsilon=epsilon,
    C=C,
    B_estim=B_estim,
    resample_each=resample_each
)
criterion = nn.CrossEntropyLoss()

for i in range(hparams['n_epochs']):
    sghmc.train_one_epoch(model, train_loader, criterion, optimizer, device='cpu', epoch=i)

# %%

model.eval()
xx, yy = torch.meshgrid(torch.linspace(-5, 5, 51), torch.linspace(-5, 5, 51))
zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

with torch.no_grad():
    logits = model.forward_snapshots(zz)

zz = logits.softmax(-1).mean(0)[:, 1]
zz = zz.view(xx.shape)

plt.title(f"Ep {i}: {hparams}")
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.contourf(xx, yy, zz, alpha=.8, zorder=-1, levels=np.linspace(0, 1, 6))
plt.colorbar()
plt.show()

# %%
