#!/bin/bash
#SBATCH --job-name=refine
#SBATCH --output=/mnt/work/dhuseljic/logs/adaptive_al/ablations/%A_%a_%x.log
#SBATCH --partition=main
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-9
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1
date;hostname
source /mnt/home/dhuseljic/envs/dal-toolbox/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/home/dhuseljic/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/work/dhuseljic/experiments/mlflow/adaptive_al/refine.db'
mlflow_exp_name='refine_v5'
# mlflow_exp_name='refine_v3_abl_depth'
# mlflow_exp_name='refine_v3_abl_alpha'
# mlflow_exp_name='refine_v3_abl_num_batches'
# mlflow_exp_name='refine_v3_abl_strategies'

datasets=(cifar10 dopanim snacks cifar100 food101 tiny_imagenet imagenet)
backbones=(dinov2 clip dinov3)
random_seeds=({1..10})

idx=$SLURM_ARRAY_TASK_ID

dataset=${datasets[0]}
backbone=${backbones[0]}
random_seed=${random_seeds[$idx]}

al_strategy=refine
depth=4
alpha=0.2
num_batches=45
init_subset_size=1000
select_strat=unc_herding

# EER stuff
eer_loss=zero_one
eer_look_ahead=mc_labels
eer_num_mc_samples=10
eer_perf_estimation=unlabeled_pool
eer_temp=False
eer_num_retraining=50
eer_ema_lmb=0.0
eer_eval_gt=True

if [ "$random_seed" -eq 1 ]; then
	python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/home/dhuseljic/projects/dal-toolbox2.0/publications/adaptive_al/
srun --mem-bind=local python main.py \
	dataset=$dataset \
	dataset.path=/mnt/work/dhuseljic/datasets \
	model.backbone=$backbone \
	al.strategy=$al_strategy \
	al.refine.progressive_depth=$depth \
	al.refine.num_batches=$num_batches \
	al.refine.alpha=$alpha \
	al.refine.init_subset_size=$init_subset_size \
	al.refine.select_strategy=$select_strat \
	al.device=cpu \
	mlflow_uri=$mlflow_uri \
	experiment_name=$mlflow_exp_name \
	random_seed=$random_seed

# al.aal.look_ahead=$eer_look_ahead \
# al.aal.num_mc_labels=$eer_num_mc_samples \
# al.aal.num_retraining_epochs=$eer_num_retraining \
# al.aal.perf_estimation=$eer_perf_estimation \
# al.aal.loss=$eer_loss \
# al.aal.temp_scaling=$eer_temp \
# al.aal.ema_lmb=$eer_ema_lmb \
# al.aal.eval_gt=$eer_eval_gt \
