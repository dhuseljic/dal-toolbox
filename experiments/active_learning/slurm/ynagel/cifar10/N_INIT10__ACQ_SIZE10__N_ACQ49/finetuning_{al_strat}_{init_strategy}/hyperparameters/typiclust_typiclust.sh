#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --array=0-324
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/stud/work/ynagel/logs/finetuning/hyperparameters/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

random_seed_array=(0 1 2)
dataset_array=(CIFAR10_PLAIN CIFAR10_NO_NORM CIFAR10_SIMCLR)
finetuning_lr_array=(0.1 0.01 0.001 0.0001 0.00001 0.000001)
linear_lr_array=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

random_seed=${random_seed_array[$((SLURM_ARRAY_TASK_ID / 108 % 3)) + 1]}

model=wideresnet2810
model_optimizer_lr=${linear_lr_array[$((SLURM_ARRAY_TASK_ID % 6)) + 1]}
model_optimizer_weight_decay=1e-4
model_train_batch_size=128
model_num_epochs=150
finetuning_lr=${finetuning_lr_array[$((SLURM_ARRAY_TASK_ID / 6 % 6)) + 1]}

dataset=${dataset_array[$((SLURM_ARRAY_TASK_ID / 36 % 3)) + 1]}

al_strat=typiclust
init_strategy=typiclust
subset_size=10000
n_init=10
acq_size=10
n_acq=49

output_dir=/mnt/stud/work/ynagel/results/finetuning_hyperparameters/${dataset}/${model}_finetuned/${al_strat}_${init_strategy}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/${model_optimizer_lr}_${finetuning_lr}/seed${random_seed}/

if [ ! -d "$output_dir" ]; then
	echo "Experiment $output_dir does not exist."
	srun python -u active_learning.py \
		model=$model \
		model.optimizer.lr=$model_optimizer_lr \
		model.optimizer.weight_decay=$model_optimizer_weight_decay \
		model.train_batch_size=$model_train_batch_size \
		model.num_epochs=$model_num_epochs \
		finetuning_lr=$finetuning_lr \
		dataset=$dataset \
		dataset_path=/mnt/stud/home/ynagel/data \
		al_strategy=$al_strat \
		al_strategy.subset_size=$subset_size \
		al_cycle.n_init=$n_init \
		al_cycle.acq_size=$acq_size \
		al_cycle.n_acq=$n_acq \
		al_cycle.init_strategy=$init_strategy \
		random_seed=$random_seed \
		output_dir=$output_dir \
		finetuning=True \
		finetuning_dir=/mnt/stud/home/ynagel/data/wide_resnet_28_10_CIFAR10_0.937.pth
else
	echo "Experiment $output_dir already run."
fi


