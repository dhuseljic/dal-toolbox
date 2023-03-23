#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=resnet18_cifar10
#SBATCH --output=/mnt/work/dhuseljic/logs/uncertainty/%x_%a.log
#SBATCH --array=1-3%3
date;hostname;pwd
source activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/uncertainty-evaluation/experiments/uncertainty/

MODEL=resnet18
DATASET=CIFAR10
OOD_DATASETS=['SVHN']
random_seed=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR=/mnt/work/dhuseljic/results/uncertainty/${DATASET}__${MODEL}/seed${random_seed}
echo "Writing results to ${OUTPUT_DIR}"

srun python -u uncertainty.py \
	model=$MODEL \
	model.optimizer.lr=0.01 \
	model.optimizer.weight_decay=0.05 \
	n_samples=100 \
	dataset=$DATASET \
	ood_datasets=$OOD_DATASETS \
	output_dir=$OUTPUT_DIR \
    random_seed=$random_seed \
	eval_interval=50
