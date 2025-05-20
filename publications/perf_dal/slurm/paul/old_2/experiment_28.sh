#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal_new/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-159%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal_2.db'
al_strategy='perf_dal_oracle'
var_sss=True

datasets=(cifar100 food101)
acq_sizes=(100 100)
subset_sizes=(1000 1000)

selection_strats=(\[random,margin,badge,alfamix,typiclust,dropquery,bait,coreset\] \[margin,badge,alfamix,typiclust,dropquery,bait,coreset\] \[badge,alfamix,typiclust,dropquery,bait,coreset\] \[alfamix,typiclust,dropquery,bait,coreset\] \[alfamix,typiclust,dropquery,bait\] \[alfamix,dropquery,bait\] \[dropquery,bait\] \[bait\])
exp_names=(experiment_23_0 experiment_23_1 experiment_23_2 experiment_23_3 experiment_23_4 experiment_23_5 experiment_23_6 experiment_23_7)

random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=${datasets[$index % 2]}
acq_size=${acq_sizes[$index % 2]}
subset_size=${subset_sizes[$index % 2]}

sel_strats=${selection_strats[$index / 2 % 8]}
mlflow_exp_name=${exp_names[$index / 2 % 8]}

random_seed=${random_seeds[$index / 16]}

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
    al.optimal.one_batch_per_strat=True \
    mlflow_uri=$mlflow_uri \
    al.device=cuda \
    experiment_name=$mlflow_exp_name \
    random_seed=$random_seed \