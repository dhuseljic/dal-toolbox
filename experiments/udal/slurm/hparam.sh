#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:4
#SBATCH --partition=main
#SBATCH --job-name=bayes_opt
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/bayes_opt/%A_%a_%x.out
#SBATCH --array=1-1
#SBATCH --exclude=gpu-v100-3
date;hostname;pwd
source activate uncertainty_evaluation
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

# model=resnet18
# model=resnet18_labelsmoothing
# model=resnet18_mixup
# model=resnet18_mcdropout
model=resnet18_ensemble 

python -u hparam_search.py \
	n_opt_samples=250 \
	model=$model \
	gpus_per_trial=0.3 \
	model.batch_size=32 \
	model.n_epochs=200 \
	budget=3456 \
	dataset=CIFAR100 \
	dataset_path=/mnt/work/dhuseljic/datasets
