#!/bin/bash
#SBATCH --job-name=adaptive_al
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/work/phahn/repositories/dal_toolbox_new/logs/adaptive_al/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-27%4

date;hostname
source /mnt/stud/work/phahn/venvs/dal-toolbox-py3104/bin/activate
export HUGGING_FACE_HUB_TOKEN=$(cat /mnt/stud/work/phahn/.huggingface_token)
ulimit -n 8192

mlflow_uri='sqlite:////mnt/stud/work/phahn/repositories/dal_toolbox_new/baselines.db'
mlflow_exp_name='baselines'

# 🧩 Define manual experiment combinations directly
manual_settings=(
    "tailor2,cifar10,dinov3,1"
    "tailor2,cifar10,dinov3,2"
    "tailor2,cifar10,dinov3,3"
    "tailor2,cifar10,dinov3,4"
    "tailor2,cifar10,dinov3,5"
    "tailor2,cifar10,dinov3,7"
    "tailor2,cifar10,dinov3,8"
    "tailor2,cifar10,dinov3,9"
    "tailor2,cifar10,dinov3,10"
    "bait,snacks,dinov3,6"
    "select_al,snacks,dinov3,2"
    "select_al,snacks,dinov3,6"
    "select_al,snacks,dinov3,8"
    "select_al,snacks,dinov3,10"
    "select_al,snacks,clip,2"
    "select_al,snacks,clip,6"
    "select_al,snacks,clip,8"
    "select_al,snacks,clip,10"
    "tailor2,cifar10,dinov2,4"
    "tailor2,cifar10,dinov2,5"
    "tailor2,cifar10,dinov2,6"
    "tailor2,cifar10,dinov2,7"
    "tailor2,cifar10,dinov2,9"
    "tailor2,cifar10,dinov2,10"
    "select_al,snacks,dinov2,2"
    "select_al,snacks,dinov2,6"
    "select_al,snacks,dinov2,8"
    "select_al,snacks,dinov2,10"
)

# pick current setting based on array ID
setting=${manual_settings[$SLURM_ARRAY_TASK_ID]}
IFS=',' read -r al_strategy dataset backbone random_seed <<< "$setting"

al_strategy=${strategies[$(( (idx / (n_seed * n_data * n_back)) % n_strat ))]}
dataset=${datasets[$(( (idx / (n_seed * n_back)) % n_data ))]}
backbone=${backbones[$(( (idx / n_seed) % n_back ))]}
random_seed=${random_seeds[$(( idx % n_seed ))]}

if [ "$random_seed" -eq 1 ]; then
        python -c "import mlflow; mlflow.set_tracking_uri(r'$mlflow_uri'); mlflow.set_experiment(r'$mlflow_exp_name')"
fi
cd /mnt/stud/work/phahn/repositories/dal_toolbox_new/dal-toolbox2.0/publications/adaptive_al/
srun python main.py \
        dataset=$dataset \
        model.backbone=$backbone \
        al.strategy=$al_strategy \
        al.device=cuda \
        mlflow_uri=$mlflow_uri \
        experiment_name=$mlflow_exp_name \
        random_seed=$random_seed \