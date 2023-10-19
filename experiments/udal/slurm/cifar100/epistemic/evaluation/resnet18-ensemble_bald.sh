#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-10%2
#SBATCH --job-name=udal
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/evaluation/%A_%a__%x.log
date;hostname;pwd
source ~/envs/dal-toolbox/bin/activate
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

dataset=CIFAR100
ood_datasets=\[CIFAR10,\ SVHN\]
model=resnet18_ensemble
al_strat=bald
n_init=2048
acq_size=2048
n_acq=9
random_seed=$SLURM_ARRAY_TASK_ID
queried_indices_json=/mnt/work/dhuseljic/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/queried_indices.json
output_dir=/mnt/work/dhuseljic/results/udal/evaluation/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

#ood_datasets=$ood_datasets \
echo "Starting script. Writing results to ${output_dir}"
srun python -u evaluate.py \
	model=resnet18 \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets \
	queried_indices_json=$queried_indices_json \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date
