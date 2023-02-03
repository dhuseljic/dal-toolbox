#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-pretrain-imagenet
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%j.log
source /mnt/stud/home/phahn/.zshrc

conda activate uncertainty_evaluation

rm -f /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/.git/index.lock

git checkout 34-pretrain-models-on-imagenet

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u experiments/pretraining/main.py \
    dataset=IMAGENET \
    dataset_path=/mnt/datasets/imagenet/ILSVRC2012/ \
    model=wideresnet2810 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_ARRAY_TASK_ID}