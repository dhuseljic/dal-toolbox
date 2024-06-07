#!/bin/bash
#SBATCH --job-name=updating
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/ssal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
#SBATCH --array=1-100

source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

DATASET=food101
OOD_DATASET=cifar10
NUM_INIT=100
NUM_NEW=100
LMB=1
GAMMA=1
LIKELIHOOD=gaussian
MODEL=laplace
NUM_EPOCHS=500
DINO_MODEL=dinov2_vitl14
SEED=$SLURM_ARRAY_TASK_ID
SCALE_FEATURES=True

mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/updating_${DATASET}

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/
srun python updating.py \
	dino_model_name=$DINO_MODEL \
	num_init_samples=$NUM_INIT \
	num_new_samples=$NUM_NEW \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	likelihood=$LIKELIHOOD \
	model.name=$MODEL \
	model.num_epochs=$NUM_EPOCHS \
	model.scale_random_features=$SCALE_FEATURES \
	mlflow_dir=$mlflow_dir \
	random_seed=$SEED \
	output_dir=/mnt/work/dhuseljic/results/ssal/test_dir \
	dataset_name=$DATASET \
	ood_dataset_name=$OOD_DATASET \
	dataset_path=/mnt/work/dhuseljic/datasets
