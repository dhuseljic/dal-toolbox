#!/bin/bash
#SBATCH --job-name=resnet_grid_search
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/grid_search/%x_%A_%a.out
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-26%3
date;hostname;pwd
source activate uncertainty_evaluation
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

model=resnet18
dataset=CIFAR10
budget=2000

# Define the range of hyperparameters to search
learning_rates=(0.001 0.01 0.1)
weight_decays=(0.0005 0.005 0.05)
random_seeds=(1 2 3)

# Get the current task index from the job array
index=$SLURM_ARRAY_TASK_ID

# Calculate the current learning rate, weight decay, and kernel scale 
# Note: bash indexing starts at 0 and zsh indexing starts at 1
learning_rate=${learning_rates[$index % 3]}
weight_decay=${weight_decays[$index / 3 % 3]}
random_seed=${random_seeds[$index / 9]}
output_dir=/mnt/work/deep_al/results/udal/hparams/${dataset}/${model}/budget${budget}/lr${learning_rate}__wd${weight_decay}/seed${random_seed}/

# Run the deep learning script with the current hyperparameters
echo "Starting script. Writing results to ${output_dir}"
echo "Learning rate: ${learning_rate}  Weight decay: ${weight_decay}  Random seed: ${random_seed}"
srun python -u train.py \
	model=$model \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets \
	budget=$budget \
	val_split=0.1 \
	output_dir=$output_dir \
	model.optimizer.lr=$learning_rate \
	model.optimizer.weight_decay=$weight_decay \
	random_seed=$random_seed
	
echo "Finished script."
date
