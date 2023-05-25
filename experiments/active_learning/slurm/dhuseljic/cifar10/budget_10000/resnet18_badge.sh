#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3%3
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/work/dhuseljic/logs/al_baselines/%A_%a__%x.log
date;hostname;pwd
source activate dal-toolbox
cd ~/projects/dal-toolbox/experiments/active_learning/

model=resnet18
dataset=CIFAR10

al_strat=badge
n_init=1000
acq_size=1000
n_acq=9
random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/work/deep_al/results/al_baselines/${dataset}/${model}/${al_strat}/budget_${budget}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.optimizer.lr=1e-3 \
	model.optimizer.weight_decay=5e-2 \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	random_seed=$random_seed \
	output_dir=$output_dir
