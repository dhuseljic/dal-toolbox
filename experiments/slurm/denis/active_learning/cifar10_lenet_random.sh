#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=al_random_lenet_cifar10
#SBATCH --output=/mnt/work/dhuseljic/logs/active_learning/%x_%a.log
#SBATCH --array=1-10%10
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uncertainty_evaluation

cd /mnt/home/dhuseljic/projects/uncertainty-evaluation/experiments/active_learning/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# ARGS
MODEL=lenet
DATASET=CIFAR10
STRATEGY=random
OOD_DATASETS=['SVHN']

N_INIT=100
ACQ_SIZE=100
N_ACQ=10

OUTPUT_DIR=/mnt/work/deep_al/results/active_learning/${DATASET}/${MODEL}/${STRATEGY}/N_INIT${N_INIT}__ACQ_SIZE${ACQ_SIZE}__N_ACQ${N_ACQ}/seed${SLURM_ARRAY_TASK_ID}/
echo "Writing results to ${OUTPUT_DIR}"

srun python -u al.py \
	model=$MODEL \
	dataset=$DATASET \
	output_dir=$OUTPUT_DIR \
    	random_seed=$SLURM_ARRAY_TASK_ID \
	al_strategy=$STRATEGY \
	al_cycle.n_init=$N_INIT \
	al_cycle.acq_size=$ACQ_SIZE \
	al_cycle.n_acq=$N_ACQ
