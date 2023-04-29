#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=0-9%10
#SBATCH --job-name=al_random_resnet18-ensemble_svhn
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/active_learning/%A_%a_%x.log
date;hostname;pwd
source activate dal-toolbox

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

model=resnet18_ensemble
dataset=SVHN
al_strat=random
n_init=100
acq_size=100
n_acq=19
random_seed=$SLURM_ARRAY_TASK_ID
init_pool_file=/mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/initial_pools/SVHN/random_${n_init}_seed${random_seed}.json
output_dir=/mnt/work/deep_al/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u active_learning.py \
	model=$model \
	model.batch_size=32 \
	model.optimizer.lr=0.06 \
	model.optimizer.weight_decay=0.001 \
	dataset=$dataset \
	ood_datasets=\[CIFAR10,\ CIFAR100\] \
	dataset_path=/mnt/work/dhuseljic/datasets \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.init_pool_file=$init_pool_file \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date
