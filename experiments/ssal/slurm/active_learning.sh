#!/bin/bash
#SBATCH --job-name=active_learning
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/ssal/active_learning/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=1-50
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

DATASET=cifar100
DINO_MODEL=dinov2_vits14

INIT_METHOD=diverse_dense
STRAT=typiclust
NUM_INIT=100
NUM_ACQ=20
ACQ_SIZE=100
SUBSET_SIZE=10000

MODEL=laplace
NUM_EPOCHS=200
SCALE_FEATURES=True
LIKELIHOOD=gaussian
LMB=1
GAMMA=10
SEED=$SLURM_ARRAY_TASK_ID

mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/al/${DATASET}

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/
srun python active_learning.py \
	dataset_name=$DATASET \
	dataset_path=/mnt/work/dhuseljic/datasets \
	dino_model_name=$DINO_MODEL \
	dino_cache_dir=/mnt/work/dhuseljic/datasets \
	al.strategy=$STRAT \
	al.init_method=$INIT_METHOD \
	al.num_init_samples=$NUM_INIT \
	al.num_acq=$NUM_ACQ \
	al.acq_size=$ACQ_SIZE \
	al.subset_size=$SUBSET_SIZE \
	model.name=$MODEL \
	model.num_epochs=$NUM_EPOCHS \
	model.scale_random_features=$SCALE_FEATURES \
	likelihood=$LIKELIHOOD \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	mlflow_dir=$mlflow_dir \
	random_seed=$SEED \
	output_dir=/mnt/work/dhuseljic/results/ssal/test_dir
