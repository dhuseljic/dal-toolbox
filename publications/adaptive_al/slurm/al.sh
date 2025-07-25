#!/bin/bash
#SBATCH --job-name=adaptive_al
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/%A_%a_%x.log
#SBATCH --ntasks=1 #SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --gres=gpu:0
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
#SBATCH --array=1-10
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/al.db'
mlflow_exp_name='baselines'
random_seed=$SLURM_ARRAY_TASK_ID
if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi

al_strategy=tcm
dataset_name=cifar10 acq_size=10 subset_size=1000
# dataset_name=stl10 acq_size=10 subset_size=Null
# dataset_name=dopanim acq_size=50 subset_size=1000
# dataset_name=snacks acq_size=20 subset_size=Null
# dataset_name=dtd acq_size=50 subset_size=Null
# dataset_name=cifar100 acq_size=100 subset_size=1000
# dataset_name=food101 acq_size=100 subset_size=1000
# dataset_name=flowers102 acq_size=100 subset_size=1000
# dataset_name=imagenet acq_size=1000 subset_size=2500

cd /mnt/home/dhuseljic/projects/dal-toolbox/publications/adaptive_al/
srun python main.py \
	dataset.name=$dataset_name \
	dataset.path=/mnt/work/dhuseljic/datasets \
	al.strategy=$al_strategy \
	al.acq_size=$acq_size \
	al.subset_size=$subset_size \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed
