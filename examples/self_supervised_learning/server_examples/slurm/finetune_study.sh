#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=finetune_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/self_supervised_learning/%x_%A_%a.log
#SBATCH --array=0-80%2

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate dal-toolbox
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/self_supervised_learning/server_examples/

models=(resnet18 wideresnet282 wideresnet2810)
datasets=(CIFAR10 CIFAR100 SVHN)
seeds=(1 2 3)
subset_sizes=(100 500 1000)

index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 3]}
dataset=${datasets[$index / 3 % 3]}
n_labeled=${subset_sizes[$index / 9 % 3]}
seed=${seeds[$index / 27 % 3]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/output/self_supervised_learning/finetuning/${dataset}/${model}/${subset_size}/seed_${seed}/
DATA_DIR=/mnt/stud/work/phahn/datasets/
MODEL_PATH=/mnt/stud/work/phahn/repositories/dal-toolbox/storage/${dataset}/${model}/seed_${seed}/pretrained_weights_seed_${seed}.pth
echo "Saving results to $OUTPUT_DIR"

srun python -u finetune.py \
    model=${model} \
    dataset=${dataset} \
    random_seed=${seed} \
    subset_size=${subset_size} \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    pretrained_weights_path=$MODEL_PATH \
    load_pretrained_weights=True \