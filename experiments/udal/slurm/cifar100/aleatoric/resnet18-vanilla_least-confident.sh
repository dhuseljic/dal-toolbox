#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-10%2
#SBATCH --job-name=udal
#SBATCH --output=/mnt/stud/home/ynagel/logs/udal/active_learning/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd /mnt/stud/home/ynagel/dal-toolbox/experiments/udal/

model=resnet18
dataset=CIFAR100
al_strat=least_confident
n_init=2048
acq_size=2048
n_acq=9
random_seed=$SLURM_ARRAY_TASK_ID
init_pool_file=/mnt/stud/home/ynagel/dal-toolbox/experiments/udal/initial_pools/${dataset}/random_${n_init}_seed${random_seed}.json
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u active_learning.py \
	model=$model \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_cycle.n_init=$n_init \
	al_cycle.init_pool_file=$init_pool_file \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date