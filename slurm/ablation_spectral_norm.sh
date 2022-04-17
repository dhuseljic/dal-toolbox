#!/usr/bin/zsh
#SBATCH --job-name=spectral-ablation
#SBATCH --array=1-120%50
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --exclude=radagast,irmo,alatar
#SBATCH --output=/mnt/work/dhuseljic/logs/grid_search_spectral//sgld_ablation_%A_%a.log
source /mnt/home/dhuseljic/.zshrc
conda activate deep_pal
eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+14)) p" /mnt/home/dhuseljic/projects/uncertainty-evaluation/slurm//ablation_spectral_norm.sh)"
exit 0

srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 100 --coeff 5 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 1000 --coeff 5 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 10 --n_samples 5000 --coeff 5 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 100 --coeff 5 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 1000 --coeff 5 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.01 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.01 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.01 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.01 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 0.1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 1 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 1 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 1 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 1 --weight_decay 0.1 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 5 --weight_decay 0 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 5 --weight_decay 0.001 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 5 --weight_decay 0.01 --device cpu 
srun python -u /mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py --n_epochs 100 --n_samples 5000 --coeff 5 --weight_decay 0.1 --device cpu 
