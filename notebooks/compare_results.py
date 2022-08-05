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

root_path_1 = Path('../results/runV2/output')
root_path_2 = Path('../results/runV3/output')
model_names = ["ensemble_5", "ensemble_10", "mcdropout", "vanilla"]
# %%
# Load all model results into one large dictionary
all_results_1 = {mn:{} for mn in model_names}
all_results_2 = {mn:{} for mn in model_names}

for all_results, root_path in zip([all_results_1, all_results_2], [root_path_1, root_path_2]):
    for model_name in model_names:
        paths = sorted(list(root_path.glob(model_name+"*")))
        for path in tqdm.tqdm(paths):
            with open(path / 'results_final.json', 'r') as f:
                results = json.load(f)
            all_results[model_name][path.stem] = {'results': results['test_history']}
# %%
# For each Experiment - For each Model - For each Metric - For each Run
# Extract the last Epochs Value of this Metric into a list
# so the structure is Dict(ModelName:Dict(MetricName:List(Values)))

test_metrics_1, test_metrics_2 = {}, {}
for all_results, test_metrics in zip([all_results_1, all_results_2], [test_metrics_1, test_metrics_2]):
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
avg_test_metrics_1, avg_test_metrics_2 = {}, {}
for avg_test_metrics, test_metrics in zip([avg_test_metrics_1, avg_test_metrics_2], [test_metrics_1, test_metrics_2]):
    for model_name in test_metrics:
        avg_test_metrics[model_name] = {}
        for metric_name in test_metrics[model_name]:
            metric_list = test_metrics[model_name][metric_name]
            avg_test_metrics[model_name][metric_name] = np.sum(metric_list)/len(metric_list)
# %%
# Bring together into one dictionary for creating a table
avg_test_metrics = {}
for model_name in model_names:
    for avg_tm, name in zip([avg_test_metrics_1, avg_test_metrics_2], ["_al", "_ub"]):
        avg_test_metrics[model_name+name]=avg_tm[model_name]
# Display results in a table
df = pd.DataFrame(avg_test_metrics).T
df

# %%
# Maybe use consistent Metrics that all should be minimized?
avg_test_metrics_new = {}
for model_name in model_names:
    for avg_tm, name in zip([avg_test_metrics_1, avg_test_metrics_2], ["_al", "_ub"]):
        new_avg_tm = {}
        for metric, value in avg_tm[model_name].items():
            if metric == 'test_acc1':
                new_avg_tm['test_err'] = 100-value
            elif metric == 'test_entropy_auroc':
                new_avg_tm['test_entropy_aoroc'] = 1-value
            elif metric == 'test_entropy_aupr':
                new_avg_tm['test_entropy_aopr'] = 1-value
            elif metric == 'test_conf_auroc':
                new_avg_tm['test_conf_aoroc'] = 1-value
            elif metric == 'test_conf_aupr':
                new_avg_tm['test_conf_aopr'] = 1-value
            else:
                new_avg_tm[metric] = value
        avg_test_metrics[model_name+name]=new_avg_tm

# Display results in a table
df = pd.DataFrame(avg_test_metrics).T
df.style.background_gradient(cmap='Blues')


# %%
# Lets also analyse the Training-Process so once again
# For each Model - For each Metric - For each Run
# Extract all Epochs Value of this Metric into a list
# so the structure is Dict(ModelName:Dict(MetricName:List(Values)))

test_metrics_1, test_metrics_2 = {}, {}
# Iteratore over each Model
for all_results, test_metrics in zip([all_results_1, all_results_2], [test_metrics_1, test_metrics_2]):
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
Metric_Dictionary_1, Metric_Dictionary_2 = {}, {}
for Metric_Dictionary, test_metrics in zip([Metric_Dictionary_1, Metric_Dictionary_2], [test_metrics_1, test_metrics_2]):
    for model_name in test_metrics:
        for metric in test_metrics[model_name]:
            if metric not in Metric_Dictionary:
                Metric_Dictionary[metric] = {}
            Metric_Dictionary[metric][model_name] = test_metrics[model_name][metric]


# %%
# Plot that Stuff
for metric_1, metric_2 in zip(Metric_Dictionary_1, Metric_Dictionary_2):
    for model_1, model_2 in zip(Metric_Dictionary_1[metric_1], Metric_Dictionary_2[metric_2]):
        plt.plot(Metric_Dictionary_1[metric_1][model_1], label=model_1+"_al")
        plt.plot(Metric_Dictionary_2[metric_2][model_2], label=model_2+"_ub")
        plt.title(metric_1)
        plt.legend()
        plt.show()
# %%