#!/bin/bash
#SBATCH --job-name=HP-experiment
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/work/dhuseljic/logs/hyperparameters/%A_%x_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-44
date;hostname;pwd
source activate dal-toolbox
cd ~/projects/dal-toolbox/experiments/hyperparameters/

# Define the range of hyperparameters 
learning_rates=(0.001 0.01 0.1)
weight_decays=(0.0005 0.005 0.05)
random_seeds=(1 2 3 4 5)

# Get the current task index from the job array
index=$SLURM_ARRAY_TASK_ID

# Calculate the current learning rate, weight decay, and kernel scale
learning_rate=${learning_rates[$index % 3]}
weight_decay=${weight_decays[$index / 3 % 3]}
random_seed=${random_seeds[$index / 9]}

dataset=CIFAR10
al_strategy=coreset
output_dir=/mnt/work/dhuseljic/results/hyperparameters/experiments/${dataset}/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}

# Run the deep learning script with the current hyperparameters
srun python -u active_learning.py \
    model.optimizer.lr=$learning_rate \
    model.optimizer.weight_decay=$weight_decay \
    dataset=$dataset \
    dataset_path=/mnt/work/dhuseljic/datasets \
    random_seed=$random_seed \
    al_strategy=$al_strategy \
    output_dir=$output_dir
