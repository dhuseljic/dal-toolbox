#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=resnet18_labelsmoothing_cifar10
#SBATCH --output=/mnt/work/dhuseljic/logs/uncertainty/%x_%a.log
#SBATCH --array=1-3%3
date;hostname;pwd
source activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/uncertainty/

model=resnet18_labelsmoothing
dataset=CIFAR10
ood_datasets=['SVHN']
random_seed=$SLURM_ARRAY_TASK_ID

for n_samples in $(seq 100 100 1000)
do
# n_samples=5000
	output_dir=/mnt/work/dhuseljic/results/uncertainty/${dataset}/${model}/${n_samples}/seed${random_seed}
	echo "Writing results to ${output_dir}"

	srun python -u uncertainty.py \
		model=$model \
		model.optimizer.lr=0.01 \
		model.optimizer.weight_decay=0.05 \
		model.label_smoothing=0.1 \
		n_samples=$n_samples \
		dataset=$dataset \
		dataset_path=/mnt/work/dhuseljic/datasets \
		ood_datasets=$ood_datasets \
		output_dir=$output_dir \
		random_seed=$random_seed \
		eval_interval=50
done
