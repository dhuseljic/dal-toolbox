# %%
from re import M
import sys
import json
import tqdm
import numpy as np
import pylab as plt
import pandas as pd

from omegaconf import OmegaConf
from pathlib import Path

root_path = Path('../results/')
# %%
# Load all model results into one large dictionary
model_names = ["ensemble", "mcdropout", "vanilla"]
all_results = {mn:{} for mn in model_names}

for model_name in model_names:
    paths = sorted(list(root_path.glob(model_name+"*")))
    for path in tqdm.tqdm(paths):
        with open(path / 'results_final.json', 'r') as f:
            results = json.load(f)
        all_results[model_name][path.stem] = {'results': results['test_history']}
# %%
# For each Model - For each Metric - For each Run
# Extract the last Epochs Value of this Metric into a list
# so the structure is Dict(ModelName:Dict(MetricName:List(Values)))

test_metrics = {}
for model_name in all_results:
    test_metrics[model_name] = {}
    model_runs_results = all_results[model_name]
    for name, exp in model_runs_results.items():
        run_results = exp['results'][-1]
        for key in run_results:
            if key not in test_metrics[model_name]:
                test_metrics[model_name][key] = []
            test_metrics[model_name][key].append(run_results[key])
# %%
# Now average these test metrics
avg_test_metrics = {}
for model_name in test_metrics:
    avg_test_metrics[model_name] = {}
    for metric_name in test_metrics[model_name]:
        metric_list = test_metrics[model_name][metric_name]
        avg_test_metrics[model_name][metric_name] = np.sum(metric_list)/len(metric_list)
# %%
# Display results in a table
pd.DataFrame(avg_test_metrics).T
# %%
# Lets also analyse the Training-Process so once again
# For each Model - For each Metric - For each Run
# Extract all Epochs Value of this Metric into a list
# so the structure is Dict(ModelName:Dict(MetricName:List(Values)))

test_metrics = {}
# Iteratore over each Model
for model_name in all_results:
    model_metrics = {}
    model_runs_results = all_results[model_name]
    # Iterate over each run
    for name, exp in model_runs_results.items():
        run_results = exp['results']
        run_metrics = {}
        # Iterate over each epoch
        for epoch, _ in enumerate(run_results):
            epoch_results = run_results[epoch]
            # Iterate over each metric
            for key in epoch_results:
                if key not in run_metrics:
                    run_metrics[key] = []
                run_metrics[key].append(epoch_results[key])
        
        # Put all Rund Metrics into one Big Model-Metrics-Dictionary with Runs x Epochs Shape per Metric
        for key in run_metrics:
            if key not in model_metrics:
                model_metrics[key] = []
            model_metrics[key].append(run_metrics[key])

        # Average them over Runs
        avg_model_metrics = {}
        for key in model_metrics:
            array = model_metrics[key]
            avg_array = np.mean(np.array(array), axis=0)
            avg_model_metrics[key] = avg_array

        # Add to the final metric dictionary
        test_metrics[model_name] = avg_model_metrics
        


# %%
# Resort all of that into a dictionary with metrics as its keys
Metric_Dictionary = {}
for model_name in test_metrics:
    for metric in test_metrics[model_name]:
        if metric not in Metric_Dictionary:
            Metric_Dictionary[metric] = {}
        Metric_Dictionary[metric][model_name] = test_metrics[model_name][metric]
# %%
# Plot that Stuff
for metric in Metric_Dictionary:
    for model in Metric_Dictionary[metric]:
        plt.plot(Metric_Dictionary[metric][model], label=model)
    plt.title(metric)
    plt.legend()
    plt.show()
# %%
