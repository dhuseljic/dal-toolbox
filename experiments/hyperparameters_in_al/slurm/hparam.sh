#!/bin/bash
#SBATCH --job-name=hparam_opt
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=12-35
#SBATCH --output=/mnt/work/dhuseljic/logs/hyperparameters/bo/%A_%a_%x.out
date;hostname;pwd
source activate dal-toolbox
cd ~/projects/dal-toolbox/experiments/hyperparameters/

# Define the range of hyperparameters 
learning_rates=(0.001 0.01 0.01 0.1)
weight_decays=(0.05 0.0005 0.05 0.005)
budgets=(2000 4000)
random_seeds=(1 2 3 4 5)

# Get the current task index from the job array
index=$SLURM_ARRAY_TASK_ID

# Calculate the current learning rate, weight decay, and kernel scale
learning_rate=${learning_rates[$index % 4]}
weight_decay=${weight_decays[$index % 4]}
budget=${budgets[$index / 4 % 2]}
random_seed=${random_seeds[$index / 8]}

al_strategy=entropy
dataset=CIFAR10
dataset_path=/mnt/work/dhuseljic/datasets
queried_indices_json=/mnt/work/dhuseljic/results/hyperparameters/experiments/${dataset}/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/queried_indices.json
output_dir=/mnt/work/dhuseljic/results/hyperparameters/BO/${dataset}/${al_strategy}/budget${budget}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/

python -u hparam.py \
    queried_indices_json=$queried_indices_json \
    output_dir=$output_dir \
    random_seed=$random_seed \
    dataset=$dataset \
    dataset_path=$dataset_path \
    num_cpus=4 \
    num_gpus=0.25 \
    budget=${budget} \
    lr=0.001 \
    weight_decay=0.05
