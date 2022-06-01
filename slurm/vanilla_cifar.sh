#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=vanilla_cifar10-vs-cifar100
#SBATCH --output=/mnt/work/dhuseljic/logs/grid_search_spectral/%A_%a.log
##SBATCH --array=1-120%50
source /mnt/home/dhuseljic/.zshrc

cd path_to_project

OUTPUT_DIR=/mnt/work/dhuseljic/uncertainty_benchmarks/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/

python main.py \
    dataset=CIFAR10_vs_CIFAR100 \
    model=vanilla \
    output_dir=./output/vanilla