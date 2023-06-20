#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3%3
#SBATCH --job-name=al_baselines_ssl
#SBATCH --output=/mnt/stud/home/ynagel/logs/al_baselines_ssl/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/

model=linear
dataset=CIFAR10

al_strat=coreset
n_init=10
acq_size=10
n_acq=5
budget=$((n_init + n_acq * acq_size))
random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/al_baselines_ssl/${dataset}/${model}/${al_strat}/budget_${budget}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.optimizer.lr=0.25 \
	model.optimizer.weight_decay=0.0 \
	model.train_batch_size=64 \
	model.num_epochs=100 \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=resnet18_cifar10_87_leacc.pth
