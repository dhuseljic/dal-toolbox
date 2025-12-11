#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perfdal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perfdal.db'
mlflow_exp_name='abl_acq_size_new'

al_strategy=perf_dal_oracle
n_bat=100
var_sss=True
sel_strats=\[alfamix,badge,bait,coreset,dropquery,dropqueryclass,margin,random,typiclass,typiclust\]

acq_sizes=(5 25 20 100 40 200)
num_acqs=(40 40 10 10 5 5)

datasets=(cifar10 dtd)
subset_sizes=(1000 Null)
num_init=(10 50)

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 2]}
subset_size=${subset_sizes[$index % 2]}
n_init=${num_init[$index % 2]}

acq_size=${acq_sizes[$index % 6]}
num_acq=${num_acqs[$index % 6]}

random_seed=${random_seeds[$index / 6]}

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
    al.optimal.num_batches=$n_bat \
    al.optimal.vary_strat_subset_size=$var_sss \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \