#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=RES-LS
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-3%3
source /mnt/stud/home/phahn/.zshrc

git checkout 40-implement-mixup-for-uncertainty-calibration

conda activate uncertainty_evaluation

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u uncertainty.py \
    dataset=CIFAR10 \
    ood_datasets=\[SVHN\] \
    model=resnet18_labelsmoothing \
    model.label_smoothing=0.05 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID}