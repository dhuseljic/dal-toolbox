#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/ssal/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --gres=gpu:0
#SBATCH --array=1-500
#SBATCH --nodelist=cpu-epyc-3

source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate

OPT=SGD
LR=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.0001), np.log10(1))))") 
WD=$(python -c "import numpy as np; print(10**(np.random.uniform(np.log10(0.0001), np.log10(1))))") 
MODEL_NAME=linear
SCALE_RANDOM_FEATURES=False
NUM_TRAIN_SAMPLES=1000

output_dir=/mnt/work/dhuseljic/results/ssal/test_dir
mlflow_dir=/mnt/work/dhuseljic/mlflow/ssal/ablation

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/ssal/

srun python ablation.py \
	optimizer.name=$OPT \
	optimizer.lr=$LR \
	optimizer.weight_decay=$WD \
	model.name=$MODEL_NAME \
	model.scale_random_features=$SCALE_RANDOM_FEATURES \
	num_train_samples=$NUM_TRAIN_SAMPLES \
	output_dir=$output_dir \
	mlflow_dir=$mlflow_dir \
	dataset_path=/mnt/work/dhuseljic/datasets
