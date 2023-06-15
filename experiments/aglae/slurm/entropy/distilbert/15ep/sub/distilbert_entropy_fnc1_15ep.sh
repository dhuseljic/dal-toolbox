#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=glae_fnc1_entropy_distilbert_15ep
#SBATCH --output=/mnt/work/lrauch/logs/aglae/%x_%a.log
#SBATCH --array=1-5%5
date;hostname;pwd
source /mnt/home/lrauch/.zshrc
conda activate uncertainty-evaluation

cd /mnt/home/lrauch/projects/dal-toolbox/experiments/aglae/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL=distilbert
DATASET=fnc1
STRATEGY=uncertainty

N_INIT=100
ACQ_SIZE=100
N_ACQ=15
GROUP=distilbert_entropy_fnc1_15ep
SEED=$SLURM_ARRAY_TASK_ID
N_EPOCHS=15

init_pool_file=~/projects/dal-toolbox/experiments/aglae/initial_pools/fnc1/random_${N_INIT}_seed${SEED}.json

OUTPUT_DIR=/mnt/work/glae/glae-results/${DATASET}/$MODEL/${STRATEGY}/15ep/sub/N_INIT${N_INIT}__ACQ_SIZE${ACQ_SIZE}__N_ACQ${N_ACQ}/seed${SLURM_ARRAY_TASK_ID}

echo "Writing results to ${OUTPUT_DIR}"

srun python -u al_txt.py \
    model=$MODEL \
    dataset=$DATASET \
    output_dir=$OUTPUT_DIR \
    random_seed=$SEED \
    al_strategy=$STRATEGY \
    al_cycle.n_init=$N_INIT \
    al_cycle.init_pool_file=$init_pool_file \
    al_cycle.acq_size=$ACQ_SIZE \
    al_cycle.n_acq=$N_ACQ \
    wandb.group=$GROUP \
    model.n_epochs=$N_EPOCHS

echo "Finished script."
date