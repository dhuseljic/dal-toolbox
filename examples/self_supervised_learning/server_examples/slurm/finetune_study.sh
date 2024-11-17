#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=finetune_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/self_supervised_learning/%x_%A_%a.log
#SBATCH --array=0-89%4

date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/self_supervised_learning/server_examples/

models=(resnet18 wideresnet282 wideresnet2810)
load_pretrained_weights=(False True)
subset_sizes=(50 100 250 500 1000)
seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 3]}
dataset=CIFAR10
load_pre=${load_pretrained_weights[$index / 3 % 2]}
subset_size=${subset_sizes[$index / 6 % 5]}
seed=${seeds[$index / 30]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/output/self_supervised_learning/finetuning/${dataset}/${model}/${load_pre}/${subset_size}/seed_${seed}/
DATA_DIR=/mnt/stud/work/phahn/datasets/
MODEL_PATH=/mnt/stud/work/phahn/repositories/dal-toolbox/storage/${dataset}/${model}/seed_1/pretrained_weights_seed_42.pth
echo "Saving results to $OUTPUT_DIR"

srun python -u finetune.py \
    model=${model} \
    dataset=${dataset} \
    random_seed=${seed} \
    subset_size=${subset_size} \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    pretrained_weights_path=$MODEL_PATH \
    load_pretrained_weights=$load_pre \