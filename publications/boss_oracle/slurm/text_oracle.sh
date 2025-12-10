#!/bin/bash
#SBATCH --job-name=baselines_boss
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/boss/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-9
#SBATCH --exclude=gpu-l40s-1
##SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/boss/bert.db'
mlflow_exp_name='bert_oracle'

datasets=(agnews dbpedia banking77 clinc)
acq_sizes=(5 10 200 200)
subset_sizes=(1000 1000 2500 2500)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

# There are 4 datasets and 10 seeds, so 40 runs in total
index=$SLURM_ARRAY_TASK_ID

al_strategy=perf_dal_oracle

num_seeds=${#random_seeds[@]}
dataset_idx=$((index / num_seeds))
seed_idx=$((index % num_seeds))

dataset_name=${datasets[$dataset_idx]}
acq_size=${acq_sizes[$dataset_idx]}
subset_size=${subset_sizes[$dataset_idx]}
random_seed=${random_seeds[$seed_idx]}

if [ $index -eq 0 ]; then
    python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

date;hostname
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/perf_dal
srun python al.py \
    dataset_name=$dataset_name \
    dataset_path=/mnt/work/dhuseljic/datasets \
    al.strategy=$al_strategy \
    al.acq_size=$acq_size \
    al.subset_size=$subset_size \
    al.device=cpu \
    optimizer.name=SGD \
    optimizer.lr=1e-3 \
    mlflow_uri=$mlflow_uri \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed
