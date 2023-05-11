#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=main
#SBATCH --job-name=uncertainty_resnet18-ls
#SBATCH --output=/mnt/work/dhuseljic/logs/uncertainty/%x_%a.log
#SBATCH --array=1-1%3
date;hostname;pwd
source activate dal-toolbox

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/uncertainty/

model=resnet18_labelsmoothing
dataset=CIFAR10
ood_datasets='[SVHN, CIFAR100]'
random_seed=$SLURM_ARRAY_TASK_ID

output_dir=/mnt/work/dhuseljic/results/uncertainty/baselines/${dataset}/${model}/seed${random_seed}
echo "Writing results to ${output_dir}"

srun python -u uncertainty.py \
	model=$model \
	model.n_epochs=200 \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets \
	"ood_datasets=$ood_datasets" \
	output_dir=$output_dir \
	random_seed=$random_seed \
	num_devices=2 \
	eval_interval=50