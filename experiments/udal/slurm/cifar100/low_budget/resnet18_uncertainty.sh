#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3%10
#SBATCH --job-name=al_uncertainty_resnet18_cifar100
#SBATCH --output=/mnt/work/dhuseljic/logs/active_learning/%A_%a_%x.log
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

model=resnet18
dataset=CIFAR100
dataset_path=/tmp/

al_strat=uncertainty
n_init=100
acq_size=100
n_acq=19

random_seed=$SLURM_ARRAY_TASK_ID

output_dir=/mnt/work/deep_al/results/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u active_learning.py \
	model=$model \
	model.n_epochs=350 \
	model.optimizer.lr=1e-2 \
	model.optimizer.weight_decay=5e-2 \
	dataset=$dataset \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	output_dir=$output_dir \
	random_seed=$random_seed
echo "Finished script."
date
