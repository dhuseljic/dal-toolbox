#!/usr/bin/zsh
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-pretraining
#SBATCH --output=/mnt/work/dhuseljic/logs/pretraining/%x_%j.log
date;hostname;pwd
source activate uncertainty_evaluation

# export NCCL_DEBUG=INFO

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/pretraining/

OUTPUT_DIR=/mnt/work/dhuseljic/results/pretraining/
echo "Saving results to $OUTPUT_DIR"

srun torchrun --standalone --nproc_per_node=2 main.py \
    dataset=CIFAR10 \
    dataset_path=/mnt/work/dhuseljic/datasets \
    model=wideresnet2810 \
    output_dir=$OUTPUT_DIR \
    random_seed=1
