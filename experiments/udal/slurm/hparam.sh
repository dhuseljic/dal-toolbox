#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:0
#SBATCH --partition=main
#SBATCH --job-name=ray_test
#SBATCH --output=/mnt/work/dhuseljic/logs/hparams/%A_%a_%x.out
#SBATCH --array=1-3%3
date;hostname;pwd
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/
source activate uncertainty_evaluation

head_node=irmo
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    srun -w $head_node ray start --num-cpus 1 --head --port=9876 --block &
    python -u hparam.py
else
    srun ray start --num-cpus $SLURM_CPUS_PER_TASK --address $head_node.ies.uni-kassel.de:9876 --block
fi

# Connect to ray master
# srun -w $head_node ray start --num-cpus $SLURM_CPUS_PER_TASK --head --port=9876 --block &
# &
# srun ray stop
