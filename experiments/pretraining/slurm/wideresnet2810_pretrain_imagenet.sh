#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=WRS2810-pretrain-imagenet
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x_%j.log
source /mnt/stud/home/phahn/.zshrc

rsync -az /mnt/datasets/imagenet/ILSVRC2012/downloads /scratch/phahn/

conda activate uncertainty_evaluation

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation/

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/
echo "Saving results to $OUTPUT_DIR"

srun python -u experiments/pretraining/main.py \
    dataset=Imagenet \
    dataset_path=/scratch/phahn/downloads/ \
    model=wideresnet2810 \
    output_dir=$OUTPUT_DIR \
    random_seed=${SLURM_JOB_ID}
