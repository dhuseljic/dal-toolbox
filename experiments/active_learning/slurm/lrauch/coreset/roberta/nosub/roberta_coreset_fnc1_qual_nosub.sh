#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=glae_fnc1_coreset_roberta_nosub
#SBATCH --output=/mnt/work/lrauch/logs/active_learning/%x_%a.log
#SBATCH --array=1-5%5
date;hostname;pwd
source /mnt/home/lrauch/.zshrc
conda activate uncertainty-evaluation

cd /mnt/home/lrauch/projects/dal-toolbox/experiments/active_learning/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL=roberta
DATASET=fnc1
STRATEGY=coreset

N_INIT=100
ACQ_SIZE=100
N_ACQ=15
GROUP=roberta_coreset_fnc1_nosub
TRAIN_SUBSET=0

OUTPUT_DIR=/mnt/work/glae/glae-results/${DATASET}/$MODEL/${STRATEGY}/nosub/N_INIT${N_INIT}__ACQ_SIZE${ACQ_SIZE}__N_ACQ${N_ACQ}/seed${SLURM_ARRAY_TASK_ID}
echo "Writing results to ${OUTPUT_DIR}"

srun python -u al_txt.py \
    model=$MODEL \
    dataset=$DATASET \
    output_dir=$OUTPUT_DIR \
    al_strategy=$STRATEGY \
    random_seed=$SLURM_ARRAY_TASK_ID \
    al_strategy=$STRATEGY \
    al_cycle.n_init=$N_INIT \
    al_cycle.acq_size=$ACQ_SIZE \
    al_cycle.n_acq=$N_ACQ \
    wandb.group=$GROUP \ 
    dataset.train_subset=TRAIN_SUBSET
