#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=main
#SBATCH --array=0-89
#SBATCH --job-name=xpal_hparams
#SBATCH --output=/mnt/stud/home/ynagel/logs/xpal_hparams/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/

random_seed_array=(0 1 2 3 4 5 6 7 8 9)
gamma_array=(0.001 0.005 0.01 0.023473 0.05 0.1 1.0 5.0 10.0)

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 9 % 10)) + 1]}
gamma=${gamma_array[$((SLURM_ARRAY_TASK_ID % 9)) + 1]}

model=pwc
dataset=CIFAR10
kernel=rbf

al_strat=random
n_init=10
acq_size=10
n_acq=9
budget=$((n_init + n_acq * acq_size))
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/xpal_hparams/${dataset}/${model}/${al_strat}/budget_${budget}/${kernel}/${gamma}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$kernel \
	model.kernel.gamma=$gamma \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/resnet18_cifar10_87_leacc.pth
