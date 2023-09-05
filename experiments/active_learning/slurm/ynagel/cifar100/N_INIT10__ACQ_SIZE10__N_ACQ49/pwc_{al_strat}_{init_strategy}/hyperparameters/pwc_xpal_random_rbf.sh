#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=main
#SBATCH --nodelist=cpu-epyc-[1-8]
#SBATCH --array=0-1200
#SBATCH --job-name=xpal_hparams
#SBATCH --output=/mnt/stud/home/ynagel/logs/xpal_hparams/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

random_seed_array=(0 1 2 3 4)
alpha_array=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14 1e-15 1e-16 1e-17 1e-18 1e-19 1e-20)
gamma_array=(calculate 0.001 0.005 0.0075 0.01 0.025 0.05 0.075 0.1 0.25 0.5 1.0 5.0 10.0) # TODO (ynagel) Reduce number of gammas based on random results

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 240 % 5)) + 1]}

model=pwc
model_kernel_name=rbf
model_kernel_gamma=${gamma_array[$((SLURM_ARRAY_TASK_ID % 12)) + 1]}

dataset=CIFAR100

al_strat=xpal
init_strategy=random
al_strat_alpha=${alpha_array[$((SLURM_ARRAY_TASK_ID / 12 % 20)) + 1]}
al_strat_kernel_name=$model_kernel_name
al_strat_kernel_gamma=$model_kernel_gamma
subset_size=10000
n_init=10
acq_size=10
n_acq=49
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/xpal_hparams/${dataset}/${model}/${al_strat}_${init_strategy}/${al_strat_alpha}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/${model_kernel_name}/${model_kernel_gamma}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$model_kernel_name \
	model.kernel.gamma=$model_kernel_gamma \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_strategy.alpha=$al_strat_alpha \
	al_strategy.kernel.name=$al_strat_kernel_name \
	al_strategy.kernel.gamma=$al_strat_kernel_gamma \
	al_strategy.subset_size=$subset_size \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/ # TODO (ynagel) Replace with new feature space data
