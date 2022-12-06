#!/bin/bash
# Ablation script that evaluates the influence of the number of training
# samples on the generalization performance and uncertainty estimates of a
# model.

# for N_SAMPLES in 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000;
for N_SAMPLES in {100..2000..100};
# for N_SAMPLES in 3000 4000 5000;
do
	echo $N_SAMPLES
	sbatch --export=ALL,N_SAMPLES=$N_SAMPLES n_samples_cifar10_resnet18.sh
	sbatch --export=ALL,N_SAMPLES=$N_SAMPLES n_samples_cifar10_resnet18-mcdropout.sh
	sbatch --export=ALL,N_SAMPLES=$N_SAMPLES n_samples_cifar10_resnet18-sngp.sh 
done
