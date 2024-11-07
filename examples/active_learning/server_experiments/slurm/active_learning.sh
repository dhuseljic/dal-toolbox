#!/bin/bash
#SBATCH --job-name=DAL-TOOLBOX
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --array=0-329%4
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/active_learning/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/active_learning/server_experiments/

# Create tupel of variables
queries=(random randomclust entropy leastconfidence margin coreset badge typiclust alfamix dropquery falcun)
datasets=(CIFAR10 CIFAR100 SVHN)
query_sizes=(10 100 100)
n_queries=(19 19 19)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
query=${queries[$index % 11]}
dset=${datasets[$index / 11 % 3]}
qs=${query_sizes[$index / 11 % 3]}
nq=${n_queries[$index / 11 % 3]}
seed=${random_seeds[$index / 33]}

# Predefine certain paths
data_dir=/mnt/stud/work/phahn/datasets/
imagenet_dir=/mnt/datasets/imagenet/ILSVRC2012/
output_dir=/mnt/stud/work/phahn/repositories/dal-toolbox/output/active_learning/baselines/dinov2/${dset}/${query}/seed_${seed}/
cache_dir=/mnt/stud/work/phahn/dino_cache/
storage_dir=/mnt/stud/work/phahn/storage/

# Run experiment
python -u active_learning.py \
        path.output_dir=$output_dir \
        path.data_dir=$data_dir \
        path.cache_dir=$cache_dir \
        path.storage_dir=$storage_dir \
        path.imagenet_dir=$imagenet_dir \
        random_seed=$seed \
        al_strategy=$query \
        al_strategy.subset_size=10000 \
        al_cycle.n_init=$qs \
        al_cycle.acq_size=$qs \
        al_cycle.n_acq=$nq \
	model=dinov2 \
        dataset=$dset \