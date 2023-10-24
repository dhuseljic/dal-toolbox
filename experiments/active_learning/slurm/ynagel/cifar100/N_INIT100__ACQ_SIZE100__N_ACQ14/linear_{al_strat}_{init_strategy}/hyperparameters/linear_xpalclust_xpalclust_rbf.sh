#!/usr/bin/zsh
#SBATCH --mem=48gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=main
#SBATCH --nodelist=cpu-epyc-6
#SBATCH --array=0-144
#SBATCH --job-name=xpal_hparams
#SBATCH --output=/mnt/stud/home/ynagel/logs/xpal_hparams/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

random_seed_array=(0 1 2 3 4)
alpha_array=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12 1e-13 1e-14 1e-15 1e-16 1e-17 1e-18 1e-19 1e-20 0.0 mean quantile_0.1 quantile_0.25 quantile_0.33 median quantile_0.66 quantile_0.75 quantile_0.9)

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 29 % 5)) + 1]}

model=linear
model_optimizer_lr=0.25
model_optimizer_weight_decay=0.0
model_train_batch_size=64
model_num_epochs=100

dataset=CIFAR100

al_strat=xpalclust
init_strategy=xpalclust
al_strat_alpha=${alpha_array[$((SLURM_ARRAY_TASK_ID % 29)) + 1]}
al_strat_kernel_name=rbf
al_strat_kernel_gamma=calculate
subset_size=10000
n_init=10
acq_size=10
n_acq=49
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/xpal_hparams/${dataset}/${model}/${al_strat}_${init_strategy}/${al_strat_alpha}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/${al_strat_kernel_name}/${al_strat_kernel_gamma}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.optimizer.lr=$model_optimizer_lr \
	model.optimizer.weight_decay=$model_optimizer_weight_decay \
	model.train_batch_size=$model_train_batch_size \
	model.num_epochs=$model_num_epochs \
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
	al_cycle.init_strategy=$init_strategy \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/wide_resnet_28_10_CIFAR100_0.682.pth
