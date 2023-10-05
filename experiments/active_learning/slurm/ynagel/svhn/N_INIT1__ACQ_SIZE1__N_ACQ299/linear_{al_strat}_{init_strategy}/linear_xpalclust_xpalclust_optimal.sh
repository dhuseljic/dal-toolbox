#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --nodelist=cpu-epyc-2
#SBATCH --array=1-10
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/stud/home/ynagel/logs/al_baselines/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

model=linear
model_optimizer_lr=0.25
model_optimizer_weight_decay=0.0
model_train_batch_size=64
model_num_epochs=100

dataset=SVHN

al_strat=xpalclust
init_strategy=xpalclust
al_strat_alpha=TODO
al_strat_kernel_name=rbf
al_strat_kernel_gamma=calculate
subset_size=10000
n_init=1
acq_size=1
n_acq=299

random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/al_baselines/${dataset}/${model}/${al_strat}_${init_strategy}_optimal/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

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
	precomputed_features_dir=/mnt/stud/home/ynagel/data/wide_resnet_28_10_SVHN_0.905.pth
