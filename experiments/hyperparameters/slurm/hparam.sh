#!/bin/bash
#SBATCH --job-name=hparam_opt
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/hyperparameters/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=256GB
#SBATCH --array=0-11%4
date;hostname;pwd
source ~/.bashrc
source activate dal-toolbox
cd /mnt/stud/work/phahn/uncertainty/dal-toolbox/experiments/hyperparameters/

# Define the range of hyperparameters 
learning_rates=(0.001 0.01 0.01 0.1)
weight_decays=(0.05 0.0005 0.05 0.005)
random_seeds=(1 2 3)

# Get the current task index from the job array
index=$SLURM_ARRAY_TASK_ID

# Calculate the current learning rate, weight decay, and kernel scale
learning_rate=${learning_rates[$index % 4]}
weight_decay=${weight_decays[$index % 4]}
random_seed=${random_seeds[$index / 4]}

al_strategy=random
budget=4000
queried_indices_json=/mnt/work/deep_al/results/hyperparameters/graphical_abstract/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/queried_indices.json
output_dir=/mnt/stud/work/phahn/uncertainty/output/hyperparameters/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/budget${budget}/

python -u hparam.py \
    queried_indices_json=$queried_indices_json \
    output_dir=$output_dir \
    random_seed=$random_seed \
    dataset_path=/mnt/stud/work/phahn/uncertainty/dal-toolbox/data \
    num_cpus=4 \
    num_gpus=0.25 \
    budget=${budget}
