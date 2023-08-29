#!/usr/bin/zsh
#SBATCH --mem=24gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --array=1-10
#SBATCH --job-name=al_baselines
#SBATCH --output=/mnt/stud/home/ynagel/logs/al_baselines/%A_%a__%x.log

date;hostname;pwd
source /mnt/stud/home/ynagel/dal-toolbox/venv/bin/activate
cd ~/dal-toolbox/experiments/active_learning/ || exit

model=linear
model_kernel_name=rbf
model_kernel_gamma=0.05
model_train_batch_size=10

dataset=SVHN

al_strat=typiclust
init_strategy=random
subset_size=10000
n_init=1
acq_size=1
n_acq=299

random_seed=$SLURM_ARRAY_TASK_ID
output_dir=/mnt/stud/home/ynagel/dal-toolbox/results/al_baselines/${dataset}/${model}/${al_strat}_${init_strategy}/N_INIT${n_init}__ACQ_SIZE${acq_size}__N_ACQ${n_acq}/seed${random_seed}/

srun python -u active_learning.py \
	model=$model \
	model.kernel.name=$model_kernel_name \
	model.kernel.gamma=$model_kernel_gamma \
	model.train_batch_size$model_train_batch_size \
	dataset=$dataset \
	dataset_path=/mnt/stud/home/ynagel/data \
	al_strategy=$al_strat \
	al_strategy.subset_size=$subset_size \
	al_cycle.n_init=$n_init \
	al_cycle.acq_size=$acq_size \
	al_cycle.n_acq=$n_acq \
	al_cycle.init_strategy=$al_strat \
	random_seed=$random_seed \
	output_dir=$output_dir \
	precomputed_features=True \
	precomputed_features_dir=/mnt/stud/home/ynagel/data/resnet50_deterministic_SVHN_0.915.pth
