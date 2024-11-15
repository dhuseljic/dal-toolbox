#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=pretrain_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/self_supervised_learning/%x.log

date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/self_supervised_learning/server_examples/

model=resnet18
dataset=CIFAR10
seed=1

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