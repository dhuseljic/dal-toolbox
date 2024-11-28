#!/bin/bash
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=finetune_study
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/semi_supervised_learning/%x_%A_%a.log
#SBATCH --array=0-11%4

date;hostname;pwd
source /mnt/stud/work/phahn/venvs/dal-toolbox/bin/activate
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/examples/semi_supervised_learning/server_examples/

algorithm=(fully_supervised pseudo_labels pi_model fixmatch)
num_epochs=(1024 1048576 1048576 1048576)
seeds=(1 2 3)

index=$SLURM_ARRAY_TASK_ID
model=resnet18
dataset=CIFAR10
alg=${algorithm[$index % 4]}
n_ep=${num_epochs[$index % 4]}
seed=${seeds[$index / 4]}

OUTPUT_DIR=/mnt/stud/work/phahn/repositories/dal-toolbox/output/semi_supervised_learning/finetuning/${dataset}/${model}/${alg}/seed_${seed}/
DATA_DIR=/mnt/stud/work/phahn/datasets/
echo "Saving results to $OUTPUT_DIR"

srun python -u main.py \
    model=$model \
    model.num_iter=$n_ep \
    dataset=$dataset \
    random_seed=${seed} \
    ssl_algorithm=$alg \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    num_labeled=200 \