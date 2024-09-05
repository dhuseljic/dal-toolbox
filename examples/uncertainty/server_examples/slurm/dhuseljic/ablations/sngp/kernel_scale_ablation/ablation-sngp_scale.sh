#!/bin/bash
# SNGP
N_SAMPLES=10000
sbatch --export=ALL,N_SAMPLES=$N_SAMPLES,KERNEL_SCALE=1 scale_cifar10_resnet18-sngp.sh
for KERNEL_SCALE in {10..200..10};
do
	sbatch --export=ALL,N_SAMPLES=$N_SAMPLES,KERNEL_SCALE=$KERNEL_SCALE scale_cifar10_resnet18-sngp.sh
done
