#!/bin/bash
#SBATCH --job-name=abl_alpha
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/ablations/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-49
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192
mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/ablations.db'
mlflow_exp_name='ablation_alpha'

datasets=(cifar10 cifar100)
backbones=(dinov2 clip)
dataset=${datasets[0]}
backbone=${backbones[0]}

alphas=(0.1 0.3 0.5 0.7 0.9)
random_seeds=({1..10})

idx=$SLURM_ARRAY_TASK_ID

# TOTAL = 5 alphas * 10 seeds = 50 experiments
n_seed=${#random_seeds[@]}
n_alpha=${#alphas[@]}

seed_idx=$(( idx % n_seed ))
alpha_idx=$(( (idx / n_seed) ))

random_seed=${random_seeds[$seed_idx]}
alpha=${alphas[$alpha_idx]}

depth=5
num_batches=100
select_strat=random

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/adaptive_al/
srun --mem-bind=local python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	model.backbone=$backbone \
	al.strategy=refine \
	al.refine.select_strategy=random \
	al.refine.progressive_depth=$depth \
	al.refine.num_batches=$num_batches \
	al.refine.alpha=$alpha \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
