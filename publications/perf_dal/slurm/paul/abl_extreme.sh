#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perfdal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perfdal.db'
mlflow_exp_name='abl_extreme_reduction'
backbone=dinov2

al_strategy=perf_dal_oracle
n_bat=50
n_ret=25
num_acq=10
var_sss=True
sel_strats=\[badge,bait,dropquery,dropqueryclass,typiclass\]

datasets=(cifar10 flowers102 dtd cifar100)
n_inits=(10 25 50 100)
acq_sizes=(20 50 100 200)
subset_sizes=(1000 Null Null 1000)

sel_strategies=(\[dropquery\] \[bait\])

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID

dataset_name=${datasets[$index % 4]}
acq_size=${acq_sizes[$index % 4]}
n_init=${n_inits[$index % 4]}
subset_size=${subset_sizes[$index % 4]}

random_seed=${random_seeds[$index / 4]}

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
    al.num_acq=$num_acq \
    al.num_init=$n_init \
    al.subset_size=$subset_size \
    al.optimal.strategies=$sel_strats \
    al.optimal.vary_strat_subset_size=$var_sss \
    al.optimal.num_batches=$n_bat \
    al.optimal.num_retraining_epochs=$n_ret \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \
    backbone=$backbone \