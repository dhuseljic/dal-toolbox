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


results = get_results(result_path, glob_pattern='CIFAR10__resnet18_sngp__[1-9]')
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

exp_names = {
    "default": "CIFAR10__resnet18_sngp__[1-9]",
    "num_inducing4096": "CIFAR10__resnet18_sngp__num_inducing4096__[1-9]",
}
data = [get_metric_dict(get_results(result_path, glob_pattern=n)) for _, n in exp_names.items()]
df = pd.DataFrame(data, index=exp_names.keys())
print(df.to_markdown())



# %%
