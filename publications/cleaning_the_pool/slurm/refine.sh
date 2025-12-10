#!/bin/bash
#SBATCH --job-name=refine
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/main/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=60-89
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/refine_new.db'
mlflow_exp_name='main_results_v1'

backbones=(dinov2 clip dinov3)
datasets=(cifar10 dopanim snacks cifar100 food101 tiny_imagenet imagenet)
random_seeds=({1..10})

# TOTAL = 3 backbones * 7 datasets * 10 seeds = 210 experiments
n_strat=${#strategies[@]}
n_data=${#datasets[@]}
n_seed=${#random_seeds[@]}
n_back=${#backbones[@]}

idx=$SLURM_ARRAY_TASK_ID

dataset=${datasets[$(( (idx / (n_seed * n_back)) % n_data ))]}
backbone=${backbones[$(( (idx / n_seed) % n_back ))]}
random_seed=${random_seeds[$(( idx % n_seed ))]}

al_strategy=refine
depth=5
alpha=0.4
num_batches=100
init_subset_size=5000
select_strat=unc_herding

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/adaptive_al/
srun --mem-bind=local python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	model.backbone=$backbone \
	al.strategy=$al_strategy \
	al.refine.progressive_depth=$depth \
	al.refine.alpha=$alpha \
	al.refine.num_batches=$num_batches \
	al.refine.init_subset_size=$init_subset_size \
	al.refine.select_strategy=$select_strat \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
