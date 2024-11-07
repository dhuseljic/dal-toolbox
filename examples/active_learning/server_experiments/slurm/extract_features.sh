#!/bin/bash
#SBATCH --job-name=DAL-TOOLBOX
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/active_learning/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/active_learning/server_experiments/

# Create tupel of variables
datasets=(CIFAR10 CIFAR100 SVHN ImageNet)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
dset=${datasets[$index]}

# Predefine certain paths
data_dir=/mnt/stud/work/phahn/datasets/
imagenet_dir=/mnt/datasets/imagenet/ILSVRC2012/
output_dir=/mnt/stud/work/phahn/repositories/dal-toolbox/output/test/{$ds}/
cache_dir=/mnt/stud/work/phahn/dino_cache/{$ds}/
storage_dir=/mnt/stud/work/phahn/storage/

# Run experiment
python -u active_learning.py \
        path.output_dir=$output_dir \
        path.data_dir=$data_dir \
        path.cache_dir=$cache_dir \
        path.storage_dir=$storage_dir \
        path.imagenet_dir=$imagenet_dir \
        random_seed=42 \
        al_strategy=random \
        al_cycle.n_acq=0 \
	model=dinov2 \
        dataset=$dset \