#!/bin/bash
#SBATCH --job-name=optimal_al
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/perf_dal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --array=1-10
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
# export HF_DATASETS_CACHE="/home/dhuseljic/.cache/huggingface/datasets"

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/perf_dal/oracle.db'
mlflow_exp_name='strategies_v2'

# dataset_name=cifar10 acq_size=10
# dataset_name=dtd acq_size=50

dataset_name=dopanim acq_size=50
# dataset_name=cifar100 acq_size=100
# dataset_name=tiny_imagenet acq_size=200
# MAYBE: dataset_name=food101 acq_size=100
backbone=dinov2

random_seed=$SLURM_ARRAY_TASK_ID

if [ "$random_seed" -eq 1 ]; then
        python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

strategies=\[random\]
#strategies=\[random,dropquery\]
#strategies=\[random,dropquery,alfamix\]
#strategies=\[random,dropquery,alfamix,typiclust\]
#strategies=\[random,dropquery,alfamix,typiclust,bait\]
#strategies=\[random,dropquery,alfamix,typiclust,bait,coreset\]
#strategies=\[random,dropquery,alfamix,typiclust,bait,coreset,margin\]
# strategies=\[random,dropquery,alfamix,typiclust,bait,coreset,margin,badge\]
# strategies=\[random,dropquery,alfamix,typiclust,bait,coreset,margin,badge,dropqueryclass\]
# strategies=\[random,dropquery,alfamix,typiclust,bait,coreset,margin,badge,dropqueryclass,typiclass\]

strategies=\[random\]
#strategies=\[random,dropquery\]
#strategies=\[random,dropquery,dropqueryclass\]
#strategies=\[random,dropquery,dropqueryclass,alfamix\]
#strategies=\[random,dropquery,dropqueryclass,alfamix,typiclust\]
#strategies=\[random,dropquery,dropqueryclass,alfamix,typiclust,bait\]
#strategies=\[random,dropquery,dropqueryclass,alfamix,typiclust,bait,coreset\]
#strategies=\[random,dropquery,dropqueryclass,alfamix,typiclust,bait,coreset,margin\]

date;hostname
cd /mnt/home/dhuseljic/projects/dal-toolbox/publications/perf_dal
srun python al.py \
        dataset_name=$dataset_name \
        dataset_path=/mnt/work/dhuseljic/datasets \
        backbone=$backbone \
        al.strategy=perf_dal_oracle \
        al.optimal.strategies=$strategies \
        al.acq_size=$acq_size \
        mlflow_uri=$mlflow_uri \
        experiment_name=$mlflow_exp_name \
        random_seed=$random_seed
