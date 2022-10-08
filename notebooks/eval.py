# %%
import json

import numpy as np
import pylab as plt
import pandas as pd

from pathlib import Path

result_path = Path('../results/')

# %%

results = []
for exp_path in result_path.glob('*'):
    exp_json = exp_path / 'results_final.json'
    with open(exp_json, 'r') as f:
        data = json.load(f)

    result_dict_final_epoch = data['test_history'][-1]
    results.append(result_dict_final_epoch )


# %%
list(exp_path.glob('*.yml'))
# %%
metrics = list(results[0].keys())
# accuracies = np.array([d['test_acc1']for d in results])

df = {}
for metric in metrics:
    value = np.mean([d[metric] for d in results])
    df[metric] = value
df = pd.DataFrame(df, index=['sngp'])
df.T

# %%
results