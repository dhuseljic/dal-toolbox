#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-16%8
#SBATCH --job-name=self_supervised_learning
#SBATCH --output=/mnt/stud/home/ynagel/logs/self_supervised_learning/CIFAR100/R18/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/self_supervised_learning/

random_seed=$SLURM_ARRAY_TASK_ID

# Grid Search
temperature_array=(0.1 0.5)
lr_array=(0.025 0.075 0.125 0.175)
accumulate_array=(1 4)

# Dataset
dataset=CIFAR100
color_distortion_strength=0.5

# SIMCLR
n_epochs=801
encoder=resnet18_deterministic
projector=mlp
projector_dim=128
temperature=${temperature_array[$((SLURM_ARRAY_TASK_ID / 8 % 2)) + 1]}
train_batch_size=512
accumulate_batches=${accumulate_array[$((SLURM_ARRAY_TASK_ID % 2)) + 1]}
optimizer_base_lr=${lr_array[$((SLURM_ARRAY_TASK_ID / 2 % 4)) + 1]}
optimizer_weight_decay=0.000001

# Linear evaluation
le_model_train_batch_size=4096
le_model_num_epochs=90
le_model_optimizer_base_lr=0.1
le_model_optimizer_weight_decay=0.0
le_model_optimizer_momentum=0.9

effective_batch_size=$((accumulate_batches * train_batch_size))
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/self_supervised_learning/${dataset}/simclr_${encoder}/baselr_${optimizer_base_lr}_bs_${effective_batch_size}_temp_${temperature}/seed${random_seed}/

srun python -u main.py \
	random_seed=$random_seed \
	dataset=$dataset \
	dataset.color_distortion_strength=$color_distortion_strength \
	dataset_path=/mnt/stud/home/ynagel/data \
	ssl_model.n_epochs=$n_epochs \
	ssl_model.encoder=$encoder \
	ssl_model.projector=$projector \
	ssl_model.projector_dim=$projector_dim \
	ssl_model.temperature=$temperature \
	ssl_model.train_batch_size=$train_batch_size \
	ssl_model.accumulate_batches=$accumulate_batches \
	ssl_model.optimizer.base_lr=$optimizer_base_lr \
	ssl_model.optimizer.weight_decay=$optimizer_weight_decay \
	le_model.train_batch_size=$le_model_train_batch_size \
	le_model.num_epochs=$le_model_num_epochs \
	le_model.optimizer.base_lr=$le_model_optimizer_base_lr \
	le_model.optimizer.weight_decay=$le_model_optimizer_weight_decay \
	le_model.optimizer.momentum=$le_model_optimizer_momentum \
	output_dir=$output_dir
	

