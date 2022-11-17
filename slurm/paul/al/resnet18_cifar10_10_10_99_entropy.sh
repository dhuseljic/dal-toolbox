#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=RES18_ENTR_CIFAR10_10_10_99
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%A_%a.log
#SBATCH --array=1-4%4
source /mnt/stud/home/phahn/.zshrc

conda activate torchal

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation

git checkout al_experiments

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u al.py \
    dataset=CIFAR10 \
    model=resnet18 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID} \
    al_cycle.n_init=10 \
    al_cycle.acq_size=10 \
    al_cycle.n_acq=99 \
    al_strategy=uncertainty 