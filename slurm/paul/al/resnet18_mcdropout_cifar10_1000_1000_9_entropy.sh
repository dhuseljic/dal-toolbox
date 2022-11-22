#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=resnet18_mcdropout_1000_1000_9
#SBATCH --output=/mnt/stud/work/phahn/uncertainty/logs/%x.log
source /mnt/stud/home/phahn/.zshrc

conda activate torchal

cd /mnt/stud/work/phahn/uncertainty/uncertainty-evaluation

git checkout al_experiments

OUTPUT_DIR=/mnt/stud/work/phahn/uncertainty/output/${SLURM_JOB_NAME}/
echo "Saving results to $OUTPUT_DIR"

srun python -u al.py \
    dataset=CIFAR10 \
    model=resnet18_mcdropout \
    model.n_passes=30 \
    output_dir=$OUTPUT_DIR \
    random_seed=1 \
    al_cycle.n_init=1000 \
    al_cycle.acq_size=1000 \
    al_cycle.n_acq=9 \
    al_strategy=uncertainty 