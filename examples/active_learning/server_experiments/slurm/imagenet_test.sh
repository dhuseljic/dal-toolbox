#!/bin/bash
#SBATCH --job-name=DAL-TOOLBOX
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --array=0-1%4
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/active_learning/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/active_learning/server_experiments/

# Create tupel of variables
queries=(random randomclust entropy leastconfidence margin coreset badge typiclust alfamix dropquery falcun)
datasets=(CIFAR10 CIFAR100 SVHN)
query_sizes=(10 100 100)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
query=random
dset=imagenet
qs=1000
seed=1

# Predefine certain paths
data_dir=/mnt/stud/work/phahn/datasets/
output_dir=/mnt/stud/work/phahn/repositories/dal-toolbox/output/baselines_new/${dset}/${query}/seed_${seed}/
cache_dir=/mnt/stud/work/phahn/dino_cache/

# Run experiment
python -u active_learning.py \
        path.output_dir=$output_dir \
        path.data_dir=$data_dir \
        path.cache_dir=$cache_dir \
        random_seed=$seed \
        al_strategy=$query \
        al_cycle.n_init=$qs \
        al_cycle.acq_size=$qs \
	model=dinov2 \
        dataset=$dset \