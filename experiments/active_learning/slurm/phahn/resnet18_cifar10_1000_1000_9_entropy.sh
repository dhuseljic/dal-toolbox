#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=entropy_1000_9
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-8%8
source /mnt/stud/home/phahn/.zshrc

conda activate torchal

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation

git checkout al_experiments

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u al.py \
    dataset=CIFAR10 \
    model=resnet18 \
    model.n_epochs=300 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID} \
    al_cycle.n_init=1000 \
    al_cycle.acq_size=1000 \
    al_cycle.n_acq=9 \
    al_strategy=uncertainty 