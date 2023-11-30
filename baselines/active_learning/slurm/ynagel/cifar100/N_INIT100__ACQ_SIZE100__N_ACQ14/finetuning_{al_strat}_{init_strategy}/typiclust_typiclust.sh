#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/stud/work/ynagel/logs/finetuning/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/baselines/active_learning/ || exit

model=wideresnet2810
model_optimizer_lr=0.1
model_optimizer_weight_decay=1e-4
model_train_batch_size=128
model_num_epochs=150
finetuning_lr=0.01

dataset=CIFAR100_SIMCLR

al_strat=typiclust
init_strategy=typiclust
subset_size=10000
n_init=100
acq_size=100
n_acq=14

random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/work/ynagel/results/active_learning/finetuning/${dataset}/${model}_finetuned/${al_strat}_${init_strategy}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

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
	finetuning_dir=/mnt/stud/home/ynagel/data/wide_resnet_28_10_CIFAR100_0.682.pth
