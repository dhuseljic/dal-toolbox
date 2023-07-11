#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=main
#SBATCH --array=0-359
#SBATCH --job-name=xpal_hparams
#SBATCH --output=/mnt/stud/home/ynagel/logs/xpal_hparams/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/

random_seed_array=(0 1 2 3 4)
alpha_array=(1e-3 1e-5 1e-7 1e-9 1e-10 1e-11 1e-12 1e-16)
gamma_array=(0.001 0.005 0.01 calculate 0.05 0.1 1.0 5.0 10.0)

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 72 % 5)) + 1]}
alpha=${alpha_array[$((SLURM_ARRAY_TASK_ID / 9 % 8)) + 1]}
gamma=${gamma_array[$((SLURM_ARRAY_TASK_ID % 9)) + 1]}

model=pwc
dataset=CIFAR10
kernel=rbf

al_strat=xpal
n_init=10
acq_size=10
n_acq=9
budget=$((n_init + n_acq * acq_size))
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/xpal_hparams/${dataset}/${model}_${alpha}_standardized/${al_strat}/budget_${budget}/${kernel}/${gamma}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$kernel \
	model.kernel.gamma=$gamma \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_strategy.alpha=$alpha \
	al_strategy.kernel.name=$kernel \
	al_strategy.kernel.gamma=$gamma \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	standardize_precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/resnet18_cifar10_87_leacc.pth
