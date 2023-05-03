# Standard
sbatch resnet18-vanilla_random.sh
sbatch resnet18-vanilla_least-confident.sh
sbatch resnet18-vanilla_margin.sh
sbatch resnet18-vanilla_entropy.sh
# Label smoothing
sbatch resnet18-labelsmoothing_random.sh
sbatch resnet18-labelsmoothing_least-confident.sh
sbatch resnet18-labelsmoothing_margin.sh
sbatch resnet18-labelsmoothing_entropy.sh
# Mixup
sbatch resnet18-mixup_random.sh
sbatch resnet18-mixup_least-confident.sh
sbatch resnet18-mixup_margin.sh
sbatch resnet18-mixup_entropy.sh
