#!/bin/bash
#SBATCH --job-name=active_learning
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/ssal/active_learning/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
#SBATCH --array=1-100

source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

DATASET=cifar10
DINO_MODEL=dinov2_vitl14

STRAT=random
NUM_INIT=10
NUM_ACQ=10
ACQ_SIZE=10

MODEL=laplace
NUM_EPOCHS=200
SCALE_FEATURES=True

LIKELIHOOD=gaussian
LMB=1
GAMMA=1
SEED=$SLURM_ARRAY_TASK_ID

mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/al_${DATASET}

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/
srun python updating.py \
	dataset_name=$DATASET \
	dataset_path=/mnt/work/dhuseljic/datasets \
	dino_model_name=$DINO_MODEL \
	al.strategy=$STRAT \
	al.num_init_samples=$NUM_INIT \
	al.num_acq=$NUM_ACQ \
	al.acq_size=$ACQ_SIZE \
	model.name=$MODEL \
	model.num_epochs=$NUM_EPOCHS \
	model.scale_random_features=$SCALE_FEATURES \
	likelihood=$LIKELIHOOD \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	mlflow_dir=$mlflow_dir \
	random_seed=$SEED \
	output_dir=/mnt/work/dhuseljic/results/ssal/test_dir