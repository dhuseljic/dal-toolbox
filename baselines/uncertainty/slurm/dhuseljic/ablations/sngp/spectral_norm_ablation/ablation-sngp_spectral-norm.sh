#!/bin/bash
# This experiment should investigate the influence of the spectral norm bound
# coefficient when 500 samples are available. So basically it is a test if the
# regularization of spectral norm is important in such low sample domains.

N_SAMPLES=500
for NORM_BOUND in 0.1 1 2 3 4 5 6 7 8 9 10 11 12;
do
	sbatch --export=ALL,N_SAMPLES=$N_SAMPLES,NORM_BOUND=$NORM_BOUND spectral_cifar10_resnet18-sngp.sh
done
