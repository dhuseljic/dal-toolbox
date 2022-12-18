#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-40-HL
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-8%8
source /mnt/stud/home/phahn/.zshrc

conda activate uncertainty_evaluation

git checkout phahn_ssl_new

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/experiments/semi_supervised_learning

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u ssl.py \
    dataset=CIFAR10 \
    model=wideresnet2810_pseudolabels \
    output_dir=$OUTPUT_DIR \
    use_hard_labels=True \
    random_seed=${SLURM_ARRAY_TASK_ID}