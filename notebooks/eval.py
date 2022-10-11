# %%
import json

import numpy as np
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf

result_path = Path('../results/')


# %%


def get_experiments(result_path, glob_pattern):
    experiments = []
    for exp_path in result_path.glob(glob_pattern):

        exp_json = exp_path / 'results_final.json'
        exp_cfg = exp_path / '.hydra' / 'config.yaml'

        with open(exp_json, 'r') as f:
            data = json.load(f)
            cfg = {}  # OmegaConf.load(exp_cfg)

        result_dict_final_epoch = data['test_history'][-1]
        experiments.append({'results': result_dict_final_epoch, 'cfg': cfg})
    return experiments


experiments = get_experiments(result_path / 'CIFAR10__resnet18', glob_pattern='seed*')
print(f'Found {len(experiments)} experiments')

# %%


def get_metric_dict(experiments, ignore_metrics=[]):
    metric_names = list(experiments[0]['results'].keys())
    d = {}
    for metric_name in metric_names:
        if metric_name in ignore_metrics:
            continue
        value = np.mean([exp['results'][metric_name] for exp in experiments])
        d[metric_name] = value
    return d


# %%
"""Results for default parameters."""

exp_names = {
    "resnet18": "CIFAR10__resnet18",
    "resnet18_sngp": "CIFAR10__resnet18_sngp",
}
ignore_metrics = ['test_SVHN_conf_auroc', 'test_SVHN_conf_aupr']

data = []
for key, name in exp_names.items():
    experiments = get_experiments(result_path / name, glob_pattern='seed*')
    metric_dict = get_metric_dict(experiments, ignore_metrics=ignore_metrics)
    data.append(metric_dict)

df = pd.DataFrame(data, index=exp_names.keys())
print(df.to_markdown())
df


# %%
"""SNGP ablation."""

exp_names = {
    "default": "CIFAR10__resnet18_sngp",
    "kernel_scale=10": "CIFAR10__resnet18_sngp__kernel_scale10",
    "num_inducing=4096": "CIFAR10__resnet18_sngp__num_inducing4096",
    "scale_features=False": "CIFAR10__resnet18_sngp__scale_features",
}
ignore_metrics = ['test_SVHN_conf_auroc', 'test_SVHN_conf_aupr']

data = []
for key, name in exp_names.items():
    experiments = get_experiments(result_path / name, glob_pattern='seed*')
    metric_dict = get_metric_dict(experiments, ignore_metrics=ignore_metrics)
    data.append(metric_dict)

df = pd.DataFrame(data, index=exp_names.keys())
print(df.to_markdown())
df


# %%
