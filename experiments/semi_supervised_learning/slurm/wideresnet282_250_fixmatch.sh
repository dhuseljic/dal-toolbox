#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=main
#SBATCH --job-name=fixmatch-250
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-2%2
source /mnt/stud/home/phahn/.zshrc

conda activate uncertainty_evaluation

git checkout 47-implement-data-distributed-parallel-for-ssl

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/experiments/semi_supervised_learning

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun torchrun --standalone --nproc_per_node=2 main.py \
    ssl_algorithm=fixmatch \
    n_labeled_samples=250 \
    output_dir=$OUTPUT_DIR \
    random_seed=$SLURM_ARRAY_TASK_ID