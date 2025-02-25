#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-139%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal_new.db'
mlflow_exp_name='image_oracle_lazy'
al_strategy=perf_dal_oracle

datasets=(cifar10 stl10 snacks flowers102 dtd food101 cifar100)
acq_sizes=(10 10 20 25 50 100 100)
subset_sizes=(2500 1000 1000 500 500 2500 2500)
num_batches=(256 512)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 7]}
acq_size=${acq_sizes[$index % 7]}
subset_size=${subset_sizes[$index % 7]}
num_bat=${num_batches[$index / 7 % 2]}
random_seed=${random_seeds[$index / 14]}

if [ $index -eq 0 ]; then
    python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

date;hostname
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/publications/perf_dal
srun python al.py \
    dataset_name=$dataset_name \
    dataset_path=/mnt/stud/work/phahn/datasets \
    al.strategy=$al_strategy \
    al.optimal.num_batches=$num_bat \
    al.optimal.strat_subset_size=$subset_size \
    al.optimal.vary_strat_subset_size=False \
    al.acq_size=$acq_size \
    al.device=cuda \
    mlflow_uri=$mlflow_uri \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed