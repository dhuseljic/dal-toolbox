#!/usr/bin/zsh
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-pretraining
#SBATCH --output=/mnt/work/dhuseljic/logs/pretraining/%x_%j.log
date;hostname;pwd
source activate uncertainty_evaluation

# export NCCL_DEBUG=INFO
rsync -avz /mnt/datasets/imagenet/ILSVRC2012 /scratch

cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/pretraining/

OUTPUT_DIR=/mnt/work/dhuseljic/results/pretraining/imagenet/
echo "Saving results to $OUTPUT_DIR"

srun torchrun --standalone --nproc_per_node=4 main.py \
    dataset=Imagenet \
    dataset_path=/scratch/ILSVRC2012 \
    model=wideresnet2810 \
    output_dir=$OUTPUT_DIR \
    random_seed=1
