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

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/perf_dal/limitations.db'
mlflow_exp_name='limitations_v1'

al_strategy=perf_dal_oracle
dataset_name=cifar10 acq_size=10 subset_size=1000 strat_subset_size=2500
# dataset_name=stl10 acq_size=10 subset_size=Null strat_subset_size=1000
# dataset_name=dopanim acq_size=50 subset_size=1000 strat_subset_size=1000
# dataset_name=snacks acq_size=20 subset_size=Null strat_subset_size=1000
# dataset_name=dtd acq_size=50 subset_size=Null strat_subset_size=400
# dataset_name=cifar100 acq_size=100 subset_size=1000 strat_subset_size=2500
# dataset_name=food101 acq_size=100 subset_size=1000 strat_subset_size=2500
# dataset_name=flowers102 acq_size=100 subset_size=1000 strat_subset_size=2500
# dataset_name=imagenet acq_size=1000 subset_size=2500 strat_subset_size=2500

## Batch Selection
num_batches=100
## Look-Ahead
look_ahead=true_labels
## Performance Estimation
perf_estimation=test_ds loss=zero_one

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
        al.optimal.num_batches=$num_batches \
        al.optimal.strat_subset_size=$strat_subset_size \
        al.optimal.look_ahead=$look_ahead \
        al.optimal.perf_estimation=$perf_estimation \
        al.optimal.loss=$loss \
        al.simulated_annealing.sa_steps=2000 \
        al.simulated_annealing.greedy_steps=400 \
        mlflow_uri=$mlflow_uri \
        experiment_name=$mlflow_exp_name \
        random_seed=$random_seed
