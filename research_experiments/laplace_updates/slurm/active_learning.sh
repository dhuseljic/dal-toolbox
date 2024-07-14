#!/bin/bash
#SBATCH --job-name=active_learning
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/pseudo_batch/active_learning/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=11-20
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HF_HOME=/mnt/work/dhuseljic/hugging_face_cache/misc
export HF_DATASETS_CACHE==/mnt/work/dhuseljic/hugging_face_cache/datasets
export TRANSFORMERS_CACHE=/mnt/work/dhuseljic/hugging_face_cache/models

DATASET=cifar10
USE_VAL=False
DINO_MODEL=dinov2_vits14

INIT_METHOD=random
STRAT=badge
NUM_INIT=10
NUM_ACQ=20
ACQ_SIZE=10
SUBSET_SIZE=1000

MODEL=laplace
NUM_EPOCHS=200
LIKELIHOOD=gaussian
LMB=1
GAMMA=15
SEED=$SLURM_ARRAY_TASK_ID

# mlflow_uri=sqlite:////mnt/work/dhuseljic/mlflow/pseudo_batch/${DATASET}_al_benchmark.db
# mlflow_uri=sqlite:////mnt/work/dhuseljic/mlflow/pseudo_batch/${DATASET}_al_ablation.db
mlflow_uri=sqlite:////mnt/home/dhuseljic/mlflow/pseudo_batch/${DATASET}_al_ablation.db
# mlflow_uri=file:////mnt/home/dhuseljic/mlflow/pseudo_batch/${DATASET}_al_ablation/
if [ "$SEED" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment('Active Learning')"
fi

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/pseudo_batch/
srun python active_learning.py \
	dataset_name=$DATASET \
	dataset_path=/mnt/work/dhuseljic/datasets \
	feature_cache_dir=/mnt/work/dhuseljic/datasets \
	use_val_split=$USE_VAL \
	dino_model_name=$DINO_MODEL \
	al.strategy=$STRAT \
	al.init_method=$INIT_METHOD \
	al.num_init_samples=$NUM_INIT \
	al.num_acq=$NUM_ACQ \
	al.acq_size=$ACQ_SIZE \
	al.subset_size=$SUBSET_SIZE \
	al.bait.grad_likelihood=binary_cross_entropy \
	model.name=$MODEL \
	model.num_epochs=$NUM_EPOCHS \
	likelihood=$LIKELIHOOD \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	mlflow_uri=$mlflow_uri \
	random_seed=$SEED \
	output_dir=/mnt/work/dhuseljic/results/pseudo_batch/test_dir
