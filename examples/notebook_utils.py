import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from dal_toolbox.models.utils.mcdropout import MCDropoutModule, ConsistentMCDropout
from dal_toolbox.models.utils.random_features import RandomFeatureGaussianProcess
from dal_toolbox.models.utils.spectral_normalization import spectral_norm_linear
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = ['DejaVu Serif']

def dataset_to_xy(dataset):
    X, y = [], []
    for batch in dataset:
        X.append(batch[0])
        y.append(batch[1])
    X = torch.stack(X).float()
    y = torch.Tensor(y).long()
    return X, y


def plot_dset(X_l, y_l, X_u=None, y_u=None, ax=None, title=None):
    # Set axis if given
    if ax:
        plt.sca(ax)

    # Either plot just labeled data or labeled and unlabeled greyed out
    if X_u != None:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=75, edgecolors='red', linewidths=2, label='labeled', zorder=5)
        plt.scatter(X_u[:, 0], X_u[:, 1], c=y_u, s=50, alpha=0.5, label='unlabeled')
    else:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=50, zorder=5, edgecolors='black')

    # Automatically generate a legend
    leg_el = scatter.legend_elements()
    # Add the "labeled" class if there are unlabeled samples present
    if X_u != None:
        leg_el[0].append(Line2D([0], [0], marker='o', color='red', markerfacecolor='white', markeredgecolor='red', markersize=10, linestyle='None'))
        leg_el[1].append("labeled")
    legend1 = plt.legend(*leg_el,
                    loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)

    # Add a title if given
    plt.title(title)

    plt.grid(visible=True)
    plt.show()
    

@torch.no_grad()
def plot_contour(model, X_l, y_l, X_u=None, y_u=None, ax=None, x_domain=3, y_domain=3, forward_mode='single'):
    model.eval()
    model.cpu()
    origin = 'lower'
    xx, yy = torch.meshgrid(torch.linspace(-x_domain, x_domain, 51), torch.linspace(-y_domain, y_domain, 51), indexing='ij')
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)

    if forward_mode == 'single':
        logits = model(zz)
        probas = logits.softmax(-1)
    elif forward_mode == 'mc':
        logits = model.mc_forward(zz)
        probas = logits.softmax(-1)
        probas = probas.mean(1)
    elif forward_mode == 'mean_field':
        logits = model(zz, mean_field=True)
        probas = logits.softmax(-1)
    else:
        raise NotImplementedError('This forward method is not taken account for. Choose one in [single, mc, mean_field]!')
        

    zz = probas[:, 1].view(xx.shape)

    if X_u != None:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=75, edgecolors='red', linewidths=2, zorder=5)
        plt.scatter(X_u[:, 0], X_u[:, 1], c=y_u, s=50, alpha=0.5, label='unlabeled')
    else:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=50, zorder=5, edgecolors='black')

    # Automatically generate a legend
    leg_el = scatter.legend_elements()
    # Add the "labeled" class if there are unlabeled samples present
    if X_u != None:
        leg_el[0].append(Line2D([0], [0], marker='o', color='red', markerfacecolor='white', markeredgecolor='red', markersize=10, linestyle='None'))
        leg_el[1].append("labeled")
    legend1 = plt.legend(*leg_el,
                    loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)

    CS = plt.contourf(xx.numpy(), yy.numpy(), zz.numpy(), alpha=.75, zorder=-1, levels=np.linspace(0, 1, 6), origin=origin)
    CS2 = plt.contour(CS, levels=[0.5], colors='black', origin=origin)
    cbar = plt.colorbar(CS)
    cbar.add_lines(CS2)

    plt.grid(visible=True)


class Net(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: int = 0, feature_dim: int = 128):
        super().__init__()

        self.first = nn.Linear(2, feature_dim)
        self.first_dropout = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(feature_dim, feature_dim)
        self.hidden_dropout = nn.Dropout(dropout_rate)
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, return_feature=False):
        x = self.act(self.first(x))
        x = self.first_dropout(x)
        x = self.act(self.hidden(x))
        x = self.hidden_dropout(x)
        out = self.last(x)
        if return_feature:
            return out, x
        return out
    
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _, indices in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits
    


class MCNet(MCDropoutModule):
    def __init__(self,
                 num_classes: int,
                 dropout_rate: int = .2,
                 feature_dim: int = 128,
                 ):
        super().__init__(n_passes=50)

        self.first = nn.Linear(2, feature_dim)
        self.first_dropout = ConsistentMCDropout(dropout_rate)
        self.hidden = nn.Linear(feature_dim, feature_dim)
        self.hidden_dropout = ConsistentMCDropout(dropout_rate)
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.first(x))
        x = self.first_dropout(x)
        x = self.act(self.hidden(x))
        x = self.hidden_dropout(x)
        out = self.last(x)
        return out

    def get_logits(self, dataloader, device):
        device='cpu'
        self.to(device)
        mc_logits_list = []
        for batch in dataloader:
            mc_logits_list.append(self.mc_forward(batch[0].to(device)).cpu())
        return torch.cat(mc_logits_list)


class SNGPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 use_spectral_norm: bool = True,
                 spectral_norm_params: dict = {},
                 gp_params: dict = {},
                 n_residual_layers: int = 6,
                 feature_dim: int = 128,
                 ):
        super().__init__()

        def linear_layer(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm_linear(nn.Linear(*args, **kwargs), **spectral_norm_params)
            else:
                return nn.Linear(*args, **kwargs)

        self.first = nn.Linear(2, feature_dim)
        self.residuals = nn.ModuleList([linear_layer(128, 128) for _ in range(n_residual_layers)])
        self.last = RandomFeatureGaussianProcess(
            in_features=feature_dim,
            out_features=num_classes,
            **gp_params,
            mc_samples=1000
        )
        self.act = nn.ReLU()
    

    def forward(self, x, mean_field=False, monte_carlo=False, return_cov=False):
        # : Added activation to first layer
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x) + x)

        if mean_field:
            out = self.last.forward_mean_field(x)
        elif monte_carlo:
            out = self.last.forward_monte_carlo(x)
        else:
            out = self.last(x, return_cov=return_cov)

        return out