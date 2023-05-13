#!/bin/bash
#SBATCH --job-name=graphical_abstract
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/work/dhuseljic/logs/hyperparameters/%A_graphical_abstract_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-8
# Array currently configured for random_seed 1 only
date;hostname;pwd
source activate dal-toolbox
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/hyperparameters/

# Define the range of hyperparameters 
learning_rates=(0.001 0.01 0.1)
weight_decays=(0.0005 0.005 0.05)
random_seeds=(1 2 3)

# Get the current task index from the job array
index=$SLURM_ARRAY_TASK_ID

# Calculate the current learning rate, weight decay, and kernel scale
learning_rate=${learning_rates[$index % 3]}
weight_decay=${weight_decays[$index / 3 % 3]}
random_seed=${random_seeds[$index / 9]}

# TODO
output_dir=/mnt/work/dhuseljic/...

# Run the deep learning script with the current hyperparameters
echo python active_learning.py \
    model.optimizer.lr=$learning_rate \
    model.optimizer.weight_decay=$weight_decay \
    random_seed=$random_seed
    