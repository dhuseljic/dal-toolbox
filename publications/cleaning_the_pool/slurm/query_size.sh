#!/bin/bash
#SBATCH --job-name=refine
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/query_sizes/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-9
##SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/refine_query_sizes.db'
mlflow_exp_name='v1'

idx=$SLURM_ARRAY_TASK_ID

backbone=dinov2
dataset=cifar100

# acq_size=10 num_acq=250
# acq_size=25 num_acq=100
# acq_size=50 num_acq=50
# acq_size=100 num_acq=25

random_seeds=({1..10})
random_seed=${random_seeds[$idx]}

al_strategy=random

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox/publications/cleaning_the_pool/
srun --mem-bind=local python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	dataset.acq_size=$acq_size \
	dataset.num_acq=$num_acq \
	model.backbone=$backbone \
	al.strategy=$al_strategy \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
