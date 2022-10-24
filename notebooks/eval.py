# %%
import json

import numpy as np
import pandas as pd

from IPython.display import display
from pathlib import Path
from omegaconf import OmegaConf

result_path = Path('../results/')


# %%


def get_experiments(result_path, glob_pattern):
    experiments = []
    for exp_path in result_path.glob(glob_pattern):

        exp_json = exp_path / 'results_final.json'
        exp_cfg = exp_path / '.hydra' / 'config.yaml'

        try:
            with open(exp_json, 'r') as f:
                data = json.load(f)
                cfg = {}  # OmegaConf.load(exp_cfg)

            result_dict_final_epoch = data['test_history'][-1]
            experiments.append({'results': result_dict_final_epoch, 'cfg': cfg})
        except:
            print(f'{exp_path} has missing results.')
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
    'deterministic': 'CIFAR10__resnet18',
    "dropout": "CIFAR10__resnet18_mcdropout",
    "sngp": "CIFAR10__resnet18_sngp",
    "sngp_optimized": "CIFAR10__resnet18_sngp__scale200",
}
ignore_metrics = ['test_SVHN_conf_auroc', 'test_SVHN_conf_aupr', 'test_SVHN_dempster_aupr', 'test_SVHN_dempster_auroc']

data = []
for key, name in exp_names.items():
    experiments = get_experiments(result_path / name, glob_pattern='seed*')
    metric_dict = get_metric_dict(experiments, ignore_metrics=ignore_metrics)
    data.append(metric_dict)

df = pd.DataFrame(data, index=exp_names.keys())
display(df)
# print(df.to_markdown())


# %%
"""SNGP kernel scale ablation."""

exp_names = {
    'deterministic': 'CIFAR10__resnet18',
    'scale10': 'ablations/CIFAR10__resnet18_sngp__scale10',
    'scale20': 'ablations/CIFAR10__resnet18_sngp__scale20',
    'scale50': 'ablations/CIFAR10__resnet18_sngp__scale50',
    'scale100': 'ablations/CIFAR10__resnet18_sngp__scale100',
    'scale200': 'ablations/CIFAR10__resnet18_sngp__scale200',
    # 'scale400': 'ablations/CIFAR10__resnet18_sngp__scale400',
}
ignore_metrics = ['test_SVHN_conf_auroc', 'test_SVHN_conf_aupr', 'test_SVHN_dempster_aupr', 'test_SVHN_dempster_auroc']

data = []
for key, name in exp_names.items():
    exp_path = result_path / name
    experiments = get_experiments(exp_path, glob_pattern='seed*')
    metric_dict = get_metric_dict(experiments, ignore_metrics=ignore_metrics)
    data.append(metric_dict)

df = pd.DataFrame(data, index=exp_names.keys())
display(df)
# print(df.to_markdown())


# %%
"""Number of Samples."""
for n_samples in [100, 500, 1000, 5000, 10000]:
    exp_names = {
        'deterministic': 'ablations/CIFAR10__resnet18__{}samples'.format(n_samples),
        'dropout': 'ablations/CIFAR10__resnet18_mcdropout__{}samples'.format(n_samples),
        'sngp': f'ablations/CIFAR10__resnet18_sngp_{n_samples}samples',
    }
    ignore_metrics = ['test_SVHN_conf_auroc', 'test_SVHN_conf_aupr', 'test_SVHN_dempster_aupr', 'test_SVHN_dempster_auroc']

    data = []
    for key, name in exp_names.items():
        exp_path = result_path /  name
        experiments = get_experiments(exp_path, glob_pattern='seed*')
        metric_dict = get_metric_dict(experiments, ignore_metrics=ignore_metrics)
        data.append(metric_dict)

    df = pd.DataFrame(data, index=exp_names.keys())
    df.columns.name = f"{n_samples} samples"
    display(df)
    print(df.to_markdown())


# %%
