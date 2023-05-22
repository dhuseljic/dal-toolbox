#!/bin/bash
#SBATCH --job-name=hparam_opt
#SBATCH --partition=main
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/hyperparameters/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-26%4
date;hostname;pwd
source /mnt/stud/home/phahn/.zshrc
cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/experiments/hyperparameters/

rm /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/.git/index.lock
git checkout hyperparameters

conda activate uncertainty_evaluation

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
queried_indices_json=   /mnt/work/deep_al/results/hyperparameters/graphical_abstract/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/queried_indices.json
ouput_dir=              /mnt/stud/work/phahn/uncertainty/output/hyperparameters/${al_strategy}/lr${learning_rate}_wd${weight_decay}/seed${random_seed}/

python -u hparam.py \
    queried_indices_json=$queried_indices_json \
    output_dir=$output_dir \
    random_seed=$random_seed \
    dataset_path=/mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/data \
    num_cpus=4 \
    num_gpus=0.5 \