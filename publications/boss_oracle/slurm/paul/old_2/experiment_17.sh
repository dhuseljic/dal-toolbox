#!/bin/bash
#SBATCH --job-name=optimal_al_baselines
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/perf_dal_new/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-69%4
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/perf_dal_2.db'
al_strategy='perf_dal_oracle'
var_sss=True
n_bat=110

mlflow_exp_names=('experiment_12_6' 'experiment_12_5' 'experiment_12_4' 'experiment_12_3' 'experiment_12_2' 'experiment_12_1' 'experiment_11_0')
selection_strategies=(\[typiclust,dropquery,bait,dropqueryclass,coreset\] \[typiclust,dropquery,bait,typiclass,dropqueryclass,coreset\] \[typiclust,dropquery,bait,typiclass,dropqueryclass,loss,coreset\] \[random,typiclust,dropquery,bait,typiclass,dropqueryclass,loss,coreset\] \[random,typiclust,dropquery,bait,typiclass,dropqueryclass,badge,loss,coreset\] \[random,typiclust,dropquery,bait,typiclass,dropqueryclass,badge,coreset,loss,alfamix\] \[random,typiclust,dropquery,bait,typiclass,dropqueryclass,margin,badge,coreset,loss,alfamix\])
random_seeds=(1 2 3 4 5 6 7 8 9 10)

index=$SLURM_ARRAY_TASK_ID
dataset_name=dopanim
acq_size=50
subset_size=1000

sel_strats=${selection_strategies[$index % 7]}
mlflow_exp_name=${mlflow_exp_names[$index % 7]}
random_seed=${random_seeds[$index / 7]}

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