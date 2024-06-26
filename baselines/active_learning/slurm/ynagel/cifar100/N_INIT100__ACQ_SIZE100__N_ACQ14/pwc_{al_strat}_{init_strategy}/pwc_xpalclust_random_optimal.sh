#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --nodelist=cpu-epyc-5
#SBATCH --array=1-10
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/stud/home/ynagel/logs/al_baselines/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/baselines/active_learning/ || exit

model=pwc
model_kernel_name=rbf
model_kernel_gamma=calculate
model_train_batch_size=10

dataset=CIFAR100

al_strat=xpalclust
init_strategy=random
al_strat_alpha=TODO
al_strat_kernel_name=$model_kernel_name
al_strat_kernel_gamma=$model_kernel_gamma
subset_size=10000
n_init=100
acq_size=100
n_acq=14

random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/al_baselines/${dataset}/${model}/${al_strat}_${init_strategy}_optimal/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$model_kernel_name \
	model.kernel.gamma=$model_kernel_gamma \
	model.train_batch_size=$model_train_batch_size \
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
