#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-ENS
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-8%8
source /mnt/stud/home/phahn/.zshrc

conda activate pytorch_alc_env

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u main.py \
    dataset=CIFAR10 \
    ood_datasets=\[SVHN\] \
    model=wideresnet2810_ensemble \
    output_dir=$OUTPUT_DIR \
    eval_interval=1 \
    n_epochs=250 \
    train_batch_size=128 \
    random_seed=${SLURM_ARRAY_TASK_ID}