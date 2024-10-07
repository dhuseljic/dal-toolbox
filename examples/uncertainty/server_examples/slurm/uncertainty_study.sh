#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=uncertainty_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/uncertainty/%x_%A_%a.log
#SBATCH --array=4-17%2

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate dal-toolbox
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/uncertainty/server_examples/

models=(deterministic labelsmoothing mixup mcdropout sngp ensemble)
seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 6]}
seed=${seeds[$index / 6 % 3]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/output/uncertainty/model_${model}/seed_${seed}/
DATA_DIR=/mnt/stud/work/phahn/datasets/
echo "Saving results to $OUTPUT_DIR"

srun python -u uncertainty.py \
    model=${model} \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    random_seed=${seed}
