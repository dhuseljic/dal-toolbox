#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=vanilla_cifar10-vs-cifar100
#SBATCH --output=/mnt/work/dhuseljic/logs/uncertainty_evaluation/%x_%A.log
source /mnt/home/dhuseljic/.zshrc

conda activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/uncertainty-evaluation/

OUTPUT_DIR=/mnt/work/dhuseljic/uncertainty_benchmarks/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u main.py \
    dataset=CIFAR10_vs_CIFAR100 \
    model=vanilla \
    output_dir=$OUTPUT_DIR \
    eval_interval=1
