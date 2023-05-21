#!/bin/bash
#SBATCH --job-name=graphical_abstract
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/work/dhuseljic/logs/hyperparameters/%A_graphical_abstract_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-0
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

al_strategy=random
queried_indices_json=/mnt/work/dhuseljic/results/hyperparameters/graphical_abstract/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/queried_indices.json
ouput_dir=/mnt/work/dhuseljic/results/hyperparameters/graphical_abstract/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/optimized/

python -u hparam.py \
    queried_indices_json=$queried_indices_json \
    random_seed=$random_seed \
    dataset_path=/mnt/work/dhuseljic/datasets \
    num_cpus=4 \
    num_gpus=0.5 \
    output_dir=$output_dir
