#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal_new_2/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal_new.db'
mlflow_exp_name='abl_sel_strats_1'
al_strategy='perf_dal_oracle'
perf_est=brier
num_retrain=25
sel_strats=\[random,typiclust,dropquery,bait,typiclass,dropqueryclass,loss,badge,coreset,alfamix\]
var_sss=True
n_bat=55

datasets=(cifar10 dtd)
acq_sizes=(10 50)
subset_sizes=(1000 Null)

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID

dataset_name=${datasets[$index % 2]}
acq_size=${acq_sizes[$index % 2]}
subset_size=${subset_sizes[$index % 2]}

random_seed=${random_seeds[$index / 2]}

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
    al.optimal.num_retraining_epochs=$num_retrain \
    al.optimal.loss=$perf_est \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \