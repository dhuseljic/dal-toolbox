# %%
import json

import numpy as np
import pylab as plt
import pandas as pd

from pathlib import Path

result_path = Path('../results/')

# %%


def get_results(result_path, glob_pattern):
    results = []
    for exp_path in result_path.glob(glob_pattern):
        exp_json = exp_path / 'results_final.json'
        with open(exp_json, 'r') as f:
            data = json.load(f)

        result_dict_final_epoch = data['test_history'][-1]
        results.append(result_dict_final_epoch)
    return results


results = get_results(result_path, glob_pattern='CIFAR10*')
results

# %%

def get_metric_dict(results):
    metrics = list(results[0].keys())
    # accuracies = np.array([d['test_acc1']for d in results])
    d = {}
    for metric in metrics:
        value = np.mean([d[metric] for d in results])
        d[metric] = value
    return d

d_normal = get_metric_dict(get_results(result_path, glob_pattern='CIFAR10*'))
d_changed = get_metric_dict(get_results(result_path, glob_pattern='CHANGED__CIFAR10*'))


pd.DataFrame([d_normal, d_changed], index=['normal', 'changed'])



# %%
