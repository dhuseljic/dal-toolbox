#!/bin/bash

sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_coreset.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_coreset_typiclust_init.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_entropy.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_entropy_typiclust_init.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_random.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_random_typiclust_init.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_typiclust.sh &
sbatch ~/dal-toolbox/experiments/active_learning/slurm/ynagel/cifar10/budget_60_ssl/linear_typiclust_typiclust_init.sh &


wait

echo "All jobs have been submitted."
