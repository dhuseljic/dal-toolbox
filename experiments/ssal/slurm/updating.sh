#!/bin/bash
#SBATCH --job-name=updating
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/ssal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=1-10

source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/updating

NUM_INIT=10
NUM_NEW=10
LMB=1
GAMMA=1
SCALE_RFF=True
NUM_EPOCHS=200
DINO_MODEL=dinov2_vitl14
SEED=$SLURM_ARRAY_TASK_ID

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/

srun python updating.py \
	dino_model_name=$DINO_MODEL \
	num_init_samples=$NUM_INIT \
	num_new_samples=$NUM_NEW \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	model.num_epochs=$NUM_EPOCHS \
	model.scale_random_features=$SCALE_RFF \
	mlflow_dir=$mlflow_dir \
	random_seed=$SEED \
	output_dir=/mnt/work/dhuseljic/results/ssal/test_dir \
	dataset_path=/mnt/work/dhuseljic/datasets
