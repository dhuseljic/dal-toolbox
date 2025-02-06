#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-809%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal.db'
mlflow_exp_name='image_baselines_v1'

query_strategies=(alfamix badge bait coreset dropquery falcun margin random typiclust)
datasets=(cifar10 stl10 dopanim snacks dtd food101 flowers102 cifar100 imagenet)
acq_sizes=(10 10 50 20 50 100 100 100 1000)
subset_sizes=(1000 Null 1000 Null Null 1000 1000 1000 2500)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
al_strategy=${query_strategies[$index % 9]}
dataset_name=${datasets[$index / 9 % 9]}
acq_size=${acq_sizes[$index / 9 % 9]}
subset_size=${subset_sizes[$index / 9 % 9]}
random_seeds=${random_seed[$index / 81]}
val_split=1

if [ $idx -eq 0 ]; then
    python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

date;hostname
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/publications/perf_dal
srun python al.py \
    dataset_name=$dataset_name \
    dataset_path=/mnt/stud/work/phahn/datasets \
    val_split=$val_split \
    al.strategy=$al_strategy \
    al.acq_size=$acq_size \
    al.subset_size=$subset_size \
    mlflow_uri=$mlflow_uri \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed