#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=RES-MIX
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-3%3
source /mnt/stud/home/phahn/.zshrc

rm /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/.git/index.lock

git checkout 40-implement-mixup-for-uncertainty-calibration

conda activate uncertainty_evaluation

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/experiments/uncertainty

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u uncertainty.py \
    dataset=CIFAR10 \
    ood_datasets=\[SVHN\] \
    model=resnet18_mixup \
    model.alpha=0.5 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID}