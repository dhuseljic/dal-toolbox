#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/bayesian_updating/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --array=1-1

# Example: Randomly pick values in the log space
LR=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.001), np.log10(0.1))))") 
WD=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.001), np.log10(1))))") 

output_dir=/mnt/work/dhuseljic/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/
mlflow_dir=/mnt/work/dhuseljic/mlflow

echo python ablation.py optimizer.name=SGD optimizer.learning_rate=$LR optimizer.weight_decay=$WD output_dir=$output_dir mlflow_dir=$mlflow_dir

