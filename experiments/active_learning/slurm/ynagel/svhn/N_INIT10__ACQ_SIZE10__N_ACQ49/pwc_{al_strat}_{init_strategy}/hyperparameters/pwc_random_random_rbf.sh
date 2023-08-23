#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=main
#SBATCH --array=0-70
#SBATCH --job-name=xpal_hparams
#SBATCH --output=/mnt/stud/home/ynagel/logs/xpal_hparams/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

random_seed_array=(0 1 2 3 4)
gamma_array=(calculate 0.001 0.005 0.0075 0.01 0.025 0.05 0.075 0.1 0.25 0.5 1.0 5.0 10.0)

model=pwc
model_kernel_name=rbf
model_kernel_gamma=${gamma_array[$((SLURM_ARRAY_TASK_ID % 14)) + 1]}
model_train_batch_size=10

dataset=SVHN

al_strat=random
init_strategy=random
subset_size=10000
n_init=10
acq_size=10
n_acq=49

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 14 % 5)) + 1]}
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/xpal_hparams/${dataset}/${model}/${al_strat}_${init_strategy}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/${model_kernel_name}/${model_kernel_gamma}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$model_kernel_name \
	model.kernel.gamma=$model_kernel_gamma \
	model.train_batch_size$model_train_batch_size \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_strategy.subset_size=$subset_size \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	al_cycle.init_strategy=$al_strat \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/ # TODO (ynagel) Replace with new feature space data
