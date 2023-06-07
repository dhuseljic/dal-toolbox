#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1
#SBATCH --job-name=ssl_al
#SBATCH --output=/mnt/stud/home/ynagel/logs/ssl_al/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/ssl_al/

model=resnet18
dataset=CIFAR10

al_strat=random
n_init=100
acq_size=100
n_acq=9
budget=$((n_init + n_acq * acq_size))
random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/ssl_al/

srun python -u ssl_al.py \
	dataset_path=/mnt/stud/home/ynagel/data \
	output_dir=$output_dir
