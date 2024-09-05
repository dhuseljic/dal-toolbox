import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
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
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=75, edgecolors='red', linewidths=2, label='labeled', zorder=5, cmap='coolwarm')
        plt.scatter(X_u[:, 0], X_u[:, 1], c=y_u, s=50, alpha=0.5, label='unlabeled', cmap='coolwarm')
    else:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=50, zorder=5, edgecolors='black', cmap='coolwarm')

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
def plot_contour(model, X_l, y_l, X_u=None, y_u=None, ax=None, x_domain=(-3, 3), y_domain=(-3, 3), forward_mode='single'):
    model.eval()
    model.cpu()
    origin = 'lower'
    xx, yy = torch.meshgrid(torch.linspace(x_domain[0], x_domain[1], 51), torch.linspace(y_domain[0], y_domain[1], 51), indexing='ij')
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

    if ax:
        plt.sca(ax)

    if X_u != None:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=75, edgecolors='red', linewidths=2, zorder=5, cmap='coolwarm')
        plt.scatter(X_u[:, 0], X_u[:, 1], c=y_u, s=50, alpha=1, label='unlabeled', edgecolors='black', linewidths=1, cmap='coolwarm')
    else:
        scatter = plt.scatter(X_l[:, 0], X_l[:, 1], c=y_l, s=50, zorder=5, edgecolors='black', cmap='coolwarm')

    # Automatically generate a legend
    leg_el = scatter.legend_elements()
    # Add the "labeled" class if there are unlabeled samples present
    if X_u != None:
        leg_el[0].append(Line2D([0], [0], marker='o', color='red', markerfacecolor='white', markeredgecolor='red', markersize=10, linestyle='None'))
        leg_el[1].append("labeled")
    legend1 = plt.legend(*leg_el,
                    loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)

    CS = plt.contourf(xx.numpy(), yy.numpy(), zz.numpy(), alpha=.75, zorder=-1, levels=np.linspace(0, 1, 6), origin=origin, cmap='coolwarm')
    CS2 = plt.contour(CS, levels=[0.5], colors='black', origin=origin)
    cbar = plt.colorbar(CS)
    cbar.add_lines(CS2)

    plt.grid(visible=True)