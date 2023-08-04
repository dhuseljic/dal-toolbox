#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3%10
#SBATCH --job-name=eval_margin_resnet18-vanilla_cifar100
#SBATCH --output=/mnt/stud/home/ynagel/logs/udal/evaluation/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd /mnt/stud/home/ynagel/dal-toolbox/experiments/udal/

dataset=CIFAR100
model=resnet18
al_strat=margin
n_init=128
acq_size=128
n_acq=38
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
	dataset_path=/mnt/stud/home/ynagel/data \
	queried_indices_json=$queried_indices_json \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date
