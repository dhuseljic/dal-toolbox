#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perfdal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-99%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perfdal.db'
mlflow_exp_name='swinv2_oracle'
backbone=swinv2

al_strategy=perf_dal_oracle
n_bat=100
var_sss=True
sel_strats=\[alfamix,badge,bait,coreset,dropquery,dropqueryclass,margin,random,typiclass,typiclust\]

datasets=(cifar10 stl10 snacks flowers102 dtd dopanim food101 cifar100 tiny_imagenet imagenet)
acq_sizes=(10 10 20 25 50 50 100 100 200 1000)
subset_sizes=(1000 Null Null Null Null 1000 1000 1000 5000 5000)

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 10]}
acq_size=${acq_sizes[$index % 10]}
subset_size=${subset_sizes[$index % 10]}

random_seed=${random_seeds[$index / 10]}

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
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \
    backbone=$backbone \