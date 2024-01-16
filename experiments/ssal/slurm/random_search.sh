#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/bayesian_updating/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=1-100

source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

# Example: Randomly pick values in the log space
LR=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.001), np.log10(0.1))))") 
WD=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.001), np.log10(1))))") 

output_dir=/mnt/work/dhuseljic/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/
mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/ablation

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/

python ablation.py optimizer.name=AdamW optimizer.lr=$LR optimizer.weight_decay=$WD dataset_path=/mnt/work/dhuseljic/datasets output_dir=$output_dir mlflow_dir=$mlflow_dir
