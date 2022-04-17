import numpy as np

# Path to python script to run.
script_path = "/mnt/home/dhuseljic/projects/uncertainty-evaluation/main.py"

# Path to save batch scripts.
batch_path = "/mnt/home/dhuseljic/projects/uncertainty-evaluation/slurm/"

# Flag whether the commands should be generated for a SLURM cluster.
use_slurm = True
python_command = "srun python -u" if use_slurm else "python -u"

# If slurm is available, this path defines where the log files are to
# be stored.
slurm_logs_path = "/mnt/work/dhuseljic/logs/grid_search_spectral/"
zshrc = "/mnt/home/dhuseljic/.zshrc"
conda_env= "deep_pal"

commands = []

n_epochs_list = [10, 100]
coeff_list = [0, 1e-2, 1e-1, 1, 5]
weight_decay_list = [0, 1e-3, 1e-2, 1e-1]
n_samples_list = [100, 1000, 5000]

file_name = f'{batch_path}/ablation_spectral_norm.sh'
n_tasks = 0
mem = 16
parallel_jobs = 50
cpus_per_task = 8

for n_epochs in n_epochs_list:
    for n_samples in n_samples_list:
        for coeff in coeff_list:
            for weight_decay in weight_decay_list:
                commands.append(
                    f"{python_command} {script_path} "
                    f"--n_epochs {n_epochs} "
                    f"--n_samples {n_samples} "
                    f"--coeff {coeff} "
                    f"--weight_decay {weight_decay} "
                    f"--device cpu "
                )
                n_tasks += 1


sbatch_config = [
    f"#!/usr/bin/zsh",
    f"#SBATCH --job-name=spectral-ablation",
    f"#SBATCH --array=1-{n_tasks}%{parallel_jobs}",
    f"#SBATCH --mem={mem}gb",
    f"#SBATCH --ntasks=1",
    f"#SBATCH --cpus-per-task={cpus_per_task}",
    f"#SBATCH --partition=main",
    f"#SBATCH --exclude=radagast,irmo,alatar",
    f"#SBATCH --output={slurm_logs_path}/sgld_ablation_%A_%a.log",
    f"source {zshrc}",
    f"conda activate {conda_env}",
    f"eval \"$(sed -n \"$(($SLURM_ARRAY_TASK_ID+14)) p\" {file_name})\"",
    f"exit 0",
]
if not use_slurm:
    sbatch_config = [sbatch_config[0]]
commands = sbatch_config + [''] + commands
with open(file_name, 'w') as f:
    for item in commands:
        f.write("%s\n" % item)
