#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=fixmatch-warmup-250
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-3%3
source /mnt/stud/home/phahn/.zshrc

conda activate uncertainty_evaluation

git checkout ssl_experiments

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u experiments/semi_supervised_learning/fixmatch.py \
    model=wideresnet282 \
    dataset=CIFAR10 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID} \
    n_labeled_samples=250 \
    ssl_algorithm.u_ratio=7