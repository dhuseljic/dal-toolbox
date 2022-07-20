#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=vanilla_cifar10-vs-cifar100
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-5%5
source /mnt/stud/home/phahn/.zshrc

conda activate pytorch_alc_env

cd /mnt/stud/work/phahn/uncertainty/uncertainty_evaluation

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u main.py \
    dataset=CIFAR10_vs_CIFAR100 \
    model=vanilla \
    output_dir=$OUTPUT_DIR \
    eval_interval=1 \
    random_seed=${SLURM_ARRAY_TASK_ID}