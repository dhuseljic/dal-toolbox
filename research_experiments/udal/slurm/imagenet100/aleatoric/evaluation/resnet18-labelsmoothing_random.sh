#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --array=1-3
#SBATCH --job-name=udal
#SBATCH --output=/mnt/stud/work/ynagel/logs/udal/%A_%a__%x.log
date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd /mnt/stud/home/ynagel/dal-toolbox/experiments/udal/

dataset=IMAGENET100
model=resnet18_labelsmoothing
al_strat=random
n_init=2048
acq_size=2048
n_acq=9
random_seed=$SLURM_ARRAY_TASK_ID
queried_indices_json=/mnt/stud/work/ynagel/results/udal/active_learning/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/queried_indices.json
output_dir=/mnt/stud/work/ynagel/results/udal/evaluation/${dataset}/${model}/${al_strat}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

echo "Starting script. Writing results to ${output_dir}"
srun python -u evaluate.py \
	model=resnet18 \
	dataset=$dataset \
	dataset_path=/mnt/datasets/imagenet/ILSVRC2012/ \
	queried_indices_json=$queried_indices_json \
	output_dir=$output_dir \
	random_seed=$random_seed 
echo "Finished script."
date
