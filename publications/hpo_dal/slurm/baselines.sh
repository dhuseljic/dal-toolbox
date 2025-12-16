#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/baselines/%A_%a_%x.log
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-1889
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/baselines.db'
mlflow_exp_name='baselines_v1'

strategies=(random margin badge bait typiclust max_herding uncertainty_herding alfamix dropquery)
backbones=(dinov2 clip dinov3)
datasets=(cifar10 dopanim snacks cifar100 food101 tiny_imagenet imagenet)
random_seeds=({1..10})

# TOTAL = 9 strat * 3 backbones * 7 datasets * 10 seeds  = 1890 experiments
n_strat=${#strategies[@]}
n_data=${#datasets[@]}
n_seed=${#random_seeds[@]}
n_back=${#backbones[@]}

idx=$SLURM_ARRAY_TASK_ID

al_strategy=${strategies[$(( (idx / (n_seed * n_back * n_data)) % n_strat ))]}
dataset=${datasets[$(( (idx / (n_seed * n_back)) % n_data ))]}
backbone=${backbones[$(( (idx / n_seed) % n_back ))]}
random_seed=${random_seeds[$(( idx % n_seed ))]}

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/adaptive_al/
srun python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	dataset.subset_size=5000 \
	model.backbone=$backbone \
	al.strategy=$al_strategy \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
