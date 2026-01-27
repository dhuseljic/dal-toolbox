#!/bin/bash
#SBATCH --job-name=refine
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/ood/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-9
##SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
export CUDA_VISIBLE_DEVICES="" # no gpus on l40s pls :)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/refine_ood.db'
mlflow_exp_name='v3'

datasets=('blood-mnist' 'derma-mnist')
strategies=(random select_al tcm tailor autoal refine uncertainty_herding)
random_seeds=({1..10})

idx=$SLURM_ARRAY_TASK_ID

backbone=dinov2
dataset=${datasets[1]}
al_strategy=${strategies[6]}

random_seed=${random_seeds[$idx]}

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox/publications/cleaning_the_pool/
srun --mem-bind=local python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	model.backbone=$backbone \
	al.strategy=$al_strategy \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
