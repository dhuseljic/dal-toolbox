#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:2
#SBATCH --partition=main
#SBATCH --job-name=bayes_opt
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/bayes_opt/%A_%a_%x.out
#SBATCH --array=1-1
#SBATCH --exclude=gpu-v100-3
date;hostname;pwd
source activate dal-toolbox
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

# model=resnet18
# model=resnet18_labelsmoothing
# model=resnet18_mixup
# model=resnet18_mcdropout
# model=resnet18_ensemble 
dataset=SVHN
budget=2000 # num_init + num_acq * acq_size

python -u hparam_search.py \
	n_opt_samples=250 \
	model=$model \
	gpus_per_trial=0.5 \
	model.batch_size=32 \
	model.n_epochs=200 \
	budget=$budget \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets
