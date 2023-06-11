#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1
#SBATCH --job-name=self_supervised_learning
#SBATCH --output=/mnt/stud/home/ynagel/logs/self_supervised_learning/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/dal-venv/bin/activate
cd ~/dal-toolbox/experiments/self_supervised_learning/


random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/self_supervised_learning/

srun python -u main.py \
	dataset_path=/mnt/stud/home/ynagel/data \
	output_dir=$output_dir
