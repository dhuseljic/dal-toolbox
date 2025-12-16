#!/bin/bash

## Replace all CAPITAL-written terms with your individual settings

#SBATCH --job-name=JOB_NAME
#SBATCH --partition=main
#SBATCH --output=LOG_FOLDER/%A_%a_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
source PATH_TO_YOUR_VENV/bin/activate

date;hostname
cd PATH_TO_DAL_TOOLBOX_FOLDER/examples
srun python active_learning.py \
    output_path=VALUE_2 \
    dataset_path=VALUE_1 \
