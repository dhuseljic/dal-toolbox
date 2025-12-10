#!/bin/bash
#SBATCH --job-name=optimal_al
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/perf_dal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
#SBATCH --array=1-10
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/perf_dal/oracle.db'
mlflow_exp_name='oracle_comparison_v4'

# dataset_name=cifar10 acq_size=10 subset_size=1000 strat_subset_size=2500
# dataset_name=stl10 acq_size=10 subset_size=Null strat_subset_size=1000
# dataset_name=snacks acq_size=20 subset_size=Null strat_subset_size=1000
# dataset_name=dopanim acq_size=50 subset_size=1000 strat_subset_size=1000
# dataset_name=dtd acq_size=50 subset_size=Null strat_subset_size=400

al_strategy=simulated_annealing_oracle

# SAS
declare -A sas_sa_steps_dict=(
        [cifar10]=250
        [stl10]=250
        [snacks]=225
        [dopanim]=215
        [dtd]=150
)
# sas_sa_steps=${sas_sa_steps_dict[$dataset_name]} sas_greedy_steps=20
sas_sa_steps=25000 sas_greedy_steps=5000

# CDO
declare -A cdo_tau_dict=(
        [cifar10]=20
        [stl10]=20
        [snacks]=10
        [dopanim]=4
        [dtd]=3
)
cdo_tau=20
#${cdo_tau_dict[$dataset_name]}

random_seed=$SLURM_ARRAY_TASK_ID

if [ "$random_seed" -eq 1 ]; then
        python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

date;hostname
cd /mnt/home/dhuseljic/projects/dal-toolbox/publications/perf_dal
srun python al.py \
        dataset_name=$dataset_name \
        dataset_path=/mnt/work/dhuseljic/datasets \
        al.strategy=$al_strategy \
        al.acq_size=$acq_size \
        al.subset_size=$subset_size \
        al.cdo.tau=$cdo_tau \
        al.sas.sa_steps=$sas_sa_steps \
        al.sas.greedy_steps=$sas_greedy_steps \
        mlflow_uri=$mlflow_uri \
        experiment_name=$mlflow_exp_name \
        random_seed=$random_seed
