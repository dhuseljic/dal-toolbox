#!/bin/bash
#SBATCH --job-name=baselines_boss
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/boss/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-319
#SBATCH --exclude=gpu-l40s-1
##SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/boss/bert.db'
mlflow_exp_name='bert_baselines_v3'

datasets=(agnews dbpedia banking77 clinc)
acq_sizes=(5 10 200 200)
subset_sizes=(1000 1000 2500 2500)
query_strategies=(alfamix badge bait coreset dropquery margin random typiclust)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 4]}
acq_size=${acq_sizes[$index % 4]}
subset_size=${subset_sizes[$index % 4]}
al_strategy=${query_strategies[$(( ($index / 4 ) % 8 ))]}
random_seed=${random_seeds[$(( $index / 32 ))]}

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
