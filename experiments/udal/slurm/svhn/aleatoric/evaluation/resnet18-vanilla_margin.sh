#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=0-9%10
#SBATCH --job-name=eval_margin_resnet18-vanilla_svhn
#SBATCH --output=/mnt/work/dhuseljic/logs/udal/active_learning/%A_%a__%x.log
date;hostname;pwd
source activate dal-toolbox
cd /mnt/home/dhuseljic/projects/dal-toolbox/experiments/udal/

dataset=SVHN
model=resnet18
al_strat=margin
n_init=100
acq_size=100
n_acq=19
random_seed=$SLURM_ARRAY_TASK_ID
queried_indices_json=/mnt/work/deep_al/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/queried_indices.json
output_dir=/mnt/work/deep_al/results/udal/evaluation/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u evaluate.py \
	model=resnet18 \
	model.batch_size=32 \
	model.optimizer.lr=0.01 \
	model.optimizer.weight_decay=0.005 \
	dataset=$dataset \
	dataset_path=/mnt/work/dhuseljic/datasets \
	queried_indices_json=$queried_indices_json \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date
