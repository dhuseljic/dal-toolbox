#!/bin/bash
#SBATCH --job-name=updating
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/pseudo_batch/updating/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4]
#SBATCH --array=1-1
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HF_HOME=/mnt/work/dhuseljic/hugging_face_cache/misc
export HF_DATASETS_CACHE==/mnt/work/dhuseljic/hugging_face_cache/datasets
export TRANSFORMERS_CACHE=/mnt/work/dhuseljic/hugging_face_cache/models

DATASET=banking77
LR=1e-1
mlflow_uri=file:////mnt/home/dhuseljic/mlflow/pseudo_batch/${DATASET}_gamma_ablation/
# mlflow_uri=file:////mnt/home/dhuseljic/mlflow/pseudo_batch/${DATASET}_initds/
# mlflow_uri=file:////mnt/home/dhuseljic/mlflow/pseudo_batch/${DATASET}_newds/

GAMMA=0.05
GAMMA_FO=0.005
GAMMA_MC=0.05
MC_SAMPLES=5

NUM_INIT=50
NUM_NEW="$(python -c 'print(str(list(range(1, 10+1, 1))).replace(" ", ""))')"

USE_VAL_SPLIT=True
LMB=1
LIKELIHOOD=categorical
NUM_EPOCHS=200
DINO_MODEL=dinov2_vits14
SEED=$SLURM_ARRAY_TASK_ID

experiment_name=updating

if [ "$SEED" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment('$experiment_name')"
fi

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/pseudo_batch/
srun python updating.py \
	dataset_name=$DATASET \
	dataset_path=/mnt/work/dhuseljic/datasets \
	use_val_split=$USE_VAL_SPLIT \
	feature_cache_dir=/mnt/work/dhuseljic/datasets \
	output_dir=/mnt/work/dhuseljic/results/test_dir \
	dino_model_name=$DINO_MODEL \
	num_init_samples=$NUM_INIT \
	num_new_samples=$NUM_NEW \
	update_lmb=$LMB \
	update_gamma=$GAMMA \
	update_gamma_fo=$GAMMA_FO \
	update_gamma_mc=$GAMMA_MC \
	likelihood=$LIKELIHOOD \
	model.num_epochs=$NUM_EPOCHS \
	model.mc_samples=$MC_SAMPLES \
	optimizer.lr=$LR \
	experiment_name=$experiment_name \
	mlflow_uri=$mlflow_uri \
	random_seed=$SEED
