#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=resnet18_mcdropout_cifar10
#SBATCH --output=/mnt/work/dhuseljic/logs/uncertainty_evaluation/n_samples/%x_%A_%a.log
#SBATCH --array=1-5%10
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/uncertainty-evaluation/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL=resnet18_mcdropout
DATASET=CIFAR10
OOD_DATASETS=['SVHN']
RESULT_DIR=/mnt/work/dhuseljic/results/uncertainty_evaluation/few_samples/
OUTPUT_DIR=$RESULT_DIR/${DATASET}/${MODEL}/${N_SAMPLES}samples/seed${SLURM_ARRAY_TASK_ID}/
echo "Writing results to ${OUTPUT_DIR}"

srun python -u uncertainty.py \
	model=$MODEL \
	dataset=$DATASET \
	ood_datasets=$OOD_DATASETS \
	output_dir=$OUTPUT_DIR \
    	random_seed=$SLURM_ARRAY_TASK_ID \
	eval_interval=50 \
	n_samples=$N_SAMPLES \
	model.optimizer.lr=0.01 \
	model.n_epochs=250
