#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/grid_search/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-1249
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/grid_search.db'
mlflow_exp_name='grid_search_v1'

backbones=(dinov2 clip dinov3)
datasets=(cifar10 dopanim snacks cifar100 food101 tiny_imagenet imagenet)

backbone=${backbones[0]}
dataset=${datasets[0]}
init_subset_size=5000

# Define the grid search parameters
alphas=(0.1 0.2 0.3 0.4 0.5)
ms=(25 50 75 100 125)
Rs=(1 2 3 4 5)
random_seeds=({1..10})

# TOTAL = 5 alphas * 5 ms * 5 Rs * 10 seeds = 1250 experiments
n_alpha=${#alphas[@]}
n_m=${#ms[@]}
n_R=${#Rs[@]}
n_seed=${#random_seeds[@]}

idx=$SLURM_ARRAY_TASK_ID

seed_idx=$(( idx % n_seed ))
R_idx=$(( (idx / n_seed) % n_R ))
m_idx=$(( (idx / (n_seed * n_R)) % n_m ))
alpha_idx=$(( (idx / (n_seed * n_R * n_m)) % n_alpha ))

al_strategy=refine
alpha=${alphas[$alpha_idx]}
num_batches=${ms[$m_idx]}
depth=${Rs[$R_idx]}
random_seed=${random_seeds[$seed_idx]}

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/adaptive_al/
echo "Running job $idx with: dataset=$dataset, backbone=$backbone, seed=$random_seed, alpha=$alpha, m=$num_batches, R=$depth"
srun --mem-bind=local python main.py \
    dataset=$dataset \
    dataset.path=/mnt/work/dhuseljic/datasets \
    model.backbone=$backbone \
    al.strategy=$al_strategy \
    al.refine.progressive_depth=$depth \
    al.refine.alpha=$alpha \
    al.refine.num_batches=$num_batches \
    al.refine.init_subset_size=$init_subset_size \
    al.device=cpu \
    mlflow_uri=$mlflow_uri \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed

