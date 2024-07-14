#!/bin/bash
# SNGP
N_SAMPLES=1000
for WD in 0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01;
do
	echo sbatch --export=ALL,N_SAMPLES=$N_SAMPLES,WD=$WD cifar10_resnet18-sngp.sh
done
