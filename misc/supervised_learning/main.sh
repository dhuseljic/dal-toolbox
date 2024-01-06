#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-1%3
#SBATCH --job-name=supervised_learning
#SBATCH --output=/mnt/stud/home/ynagel/logs/supervised_learning/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/supervised_learning/

model=resnet18
epochs=200
learning_rate=0.1
weight_decay=2e-4
batch_size=128
norm_bound=1.0
n_power_iterations=1

dataset=SVHN
random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/supervised_learning/${dataset}/${model}/seed${random_seed}/

srun python -u supervised_learning.py \
	model=$model \
	model.num_epochs=$epochs \
	model.train_batch_size=$batch_size \
	model.norm_bound=$norm_bound \
	model.n_power_iterations=$n_power_iterations \
	model.optimizer.lr=$learning_rate \
	model.optimizer.weight_decay=$weight_decay \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	random_seed=$random_seed \
	output_dir=$output_dir
