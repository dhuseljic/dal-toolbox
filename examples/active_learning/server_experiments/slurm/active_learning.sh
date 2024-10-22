#!/bin/bash
#SBATCH --job-name=DAL-TOOLBOX
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --array=0-35%4
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/active_learning/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/active_learning/server_experiments/

# Create tupel of variables
queries=(random randomclust entropy leastconfidence margin coreset badge typiclust alfamix dropquery falcun cal)
random_seeds=(1 2 3)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
query=${queries[$index % 12]}
seed=${random_seeds[$index / 12]}

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
output_dir=/mnt/stud/work/phahn/repositories/dal-toolbox/output/baselines_1/${query}/seed_${seed}/
cache_dir=/mnt/stud/work/phahn/repositories/OptDal/cache/

# Run experiment
python -u active_learning.py \
        path.output_dir=$output_dir \
        path.data_dir=$dataset_path \
        path.cache_dir=$cache_dir \
        random_seed=$seed \
        query_strategy=$query \
