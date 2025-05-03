#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal_new/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-79%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal_2.db'
mlflow_exp_name='experiment_14'
al_strategy='perf_dal_oracle'
sel_strats=\[typiclust,dropquery,bait,dropqueryclass,coreset,typiclass,loss,badge,random,marginsampling,alfamix\]
var_sss=True

datasets=(cifar10 dtd)
acq_sizes=(10 50)
subset_sizes=(1000 Null)

num_batches=(220 160 82 55)
retraining_epochs=(25 37 75 100)

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 2]}
acq_size=${acq_sizes[$index % 2]}
subset_size=${subset_sizes[$index % 2]}

n_bat=${num_batches[$index / 2 % 4]}
n_ret=${retraining_epochs[$index / 2 % 4]}

random_seed=${random_seeds[$index / 8]}

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
    al.optimal.strategies=$sel_strats \
    al.optimal.vary_strat_subset_size=$var_sss \
    al.optimal.num_batches=$n_bat \
    al.optimal.num_retraining_epochs=$n_ret \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \