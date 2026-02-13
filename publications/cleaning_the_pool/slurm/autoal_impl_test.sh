#!/bin/bash
#SBATCH --job-name=adaptive_al
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal-toolbox/logs/adaptive_al/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:0
#SBATCH --array=0-89
#SBATCH --exclude=gpu-a100-[1-5],gpu-v100-[1-4],gpu-l40s-1

date;hostname
source /mnt/stud/work/phahn/venvs/dal-toolbox-py3104/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/stud/work/phahn/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal-toolbox/test.db'
mlflow_exp_name='autoal_test'

strategies=(random autoal uncertainty_herding)
datasets=(cifar10 dopanim cifar100)
random_seeds=(1 2 3 4 5 6 7 8 9 10)

idx=$SLURM_ARRAY_TASK_ID
al_strategy=${strategies[$idx % 3]}
dataset=${datasets[$idx / 3 % 3]}
random_seed=${random_seeds[$idx / 9]}

backbone=dinov3

if [ "$random_seed" -eq 1 ]; then
        python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/stud/work/phahn/repositories/dal-toolbox/dal-toolbox/publications/cleaning_the_pool/
srun python main.py \
        dataset=$dataset \
        dataset.path=/mnt/stud/work/phahn/datasets/
        model.backbone=$backbone \
        al.strategy=$al_strategy \
        mlflow_uri=$mlflow_uri \
        experiment_name=$mlflow_exp_name \
        random_seed=$random_seed \