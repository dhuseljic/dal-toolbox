#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perfdal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-95%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perfdal.db'
mlflow_exp_name='dinov2_finetune_hps_2'
backbone=dinov2

lrs_backbone=(1e-4 1e-5 1e-6 1e-7)
wds_backbone=(1e-4 1e-5 1e-6 1e-7)
apply_train_transform=(False True)
random_seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
dataset_name=cifar10
acq_size=10
subset_size=1000
al_strategy=random
lr=0.01
wd=0.0005

lr_back=${lrs_backbone[$index / 4]}
wd_back=${wds_backbone[$index / 4 % 4]}
apply_tt=${apply_train_transform[$index / 16 % 2]}
random_seed=${random_seeds[$index / 32]}

if [ $index -eq 0 ]; then
    python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

date;hostname
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/publications/perf_dal
srun python al.py \
    dataset_name=$dataset_name \
    dataset_path=/mnt/stud/work/phahn/datasets \
    al.strategy=$al_strategy \
    al.acq_size=$acq_size \
    al.subset_size=$subset_size \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \
    backbone=$backbone \
    finetune_backbone=True \
    optimizer.lr=$lr \
    optimizer.weight_decay=$wd \
    optimizer.lr_backbone=$lr_back \
    optimizer.weight_decay_backbone=$wd_back \
    apply_train_transforms=$apply_tt \