#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=pretrain_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/self_supervised_learning/%x_%A_%a.log
#SBATCH --array=0-26%2

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate dal-toolbox
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/self_supervised_learning/server_examples/

models=(resnet18 wideresnet282 wideresnet2810)
datasets=(CIFAR10 CIFAR100 SVHN)
seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 3]}
dataset=${datasets[$index / 3 % 3]}
seed=${seeds[$index / 9 % 3]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/output/self_supervised_learning/pretraining/${dataset}/${model}/seed_${seed}/
DATA_DIR=/mnt/stud/work/phahn/datasets/
MODEL_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/storage/${dataset}/${model}/seed_${seed}/
echo "Saving results to $OUTPUT_DIR"

srun python -u pretrain.py \
    model=${model} \
    dataset=${dataset} \
    random_seed=${seed} \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    model_dir=$MODEL_DIR \