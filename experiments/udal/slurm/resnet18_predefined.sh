#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3%10
#SBATCH --job-name=al_predefined_resnet18_cifar10
#SBATCH --output=/mnt/work/dhuseljic/logs/active_learning/%A_%a_%x.log
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/active_learning/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

model=resnet18
al_strat=predefined
dataset=CIFAR10
dataset_path=/tmp/

n_init=1000
acq_size=1000
n_acq=9
random_seed=$SLURM_ARRAY_TASK_ID

# Define the queried indices file from another experiment
model_predefined=resnet18
al_start_predefined=random
queried_indices_json=/mnt/work/deep_al/results/active_learning/${dataset}/${model_predefined}/${al_start_predefined}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/queried_indices.json

output_dir=/mnt/work/deep_al/results/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/${model_predefined}_${al_start_predefined}_queries/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u al.py \
	model=$model \
	model.optimizer.lr=1e-2 \
	dataset=$dataset \
	al_strategy=$al_strat \
	al_strategy.queried_indices_json=$queried_indices_json \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	output_dir=$output_dir \
	random_seed=$random_seed
echo "Finished script."
date
