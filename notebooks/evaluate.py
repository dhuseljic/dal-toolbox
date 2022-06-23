"""Notebook for evaluation of models."""
# %%
# fmt: off
import sys
import json
import math

sys.path.append('../')

import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from pathlib import Path

# fmt: on

# %%


def load_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return data


def get_metrics_from_jsons(paths, metric_keys, split='test', type='last'):
    results = {key: [] for key in metric_keys}

    for json_file in paths:
        data = load_json(json_file)
        history = data['test_history'] if split == 'test' else data['train_history']

        for key in metric_keys:
            metric_list = [d[key] for d in history]
            if type == 'last':
                val = metric_list[-1]
            elif type == 'best':
                val = np.max(metric_list)
            else:
                raise NotImplementedError
            results[key].append(val)
    return results


# %%
results_dir = Path('/mnt/work/dhuseljic/uncertainty_evaluation/')
# %%
"""Vanilla"""
results = list(results_dir.glob('*vanilla*'))
result_jsons = [list(path.glob('*.json'))[0] for path in results if list(path.glob('*.json'))]

res = get_metrics_from_jsons(result_jsons, ['test_acc1', 'test_auroc'])
print('Found {} experiments.'.format(len(result_jsons)))
print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(res['test_acc1']), np.std(res['test_acc1'])))
print('AUROC : {:.3f} ± {:.3f}'.format(np.mean(res['test_auroc']), np.std(res['test_auroc'])))

# %%
"""SNGP"""
results = list(results_dir.glob('*sngp*'))
result_jsons = [list(path.glob('*.json'))[0] for path in results if list(path.glob('*.json'))]
res = get_metrics_from_jsons(result_jsons, ['test_acc1', 'test_auroc'])

print('Found {} experiments.'.format(len(result_jsons)))
print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(res['test_acc1']), np.std(res['test_acc1'])))
print('AUROC : {:.3f} ± {:.3f}'.format(np.mean(res['test_auroc']), np.std(res['test_auroc'])))

# %%
