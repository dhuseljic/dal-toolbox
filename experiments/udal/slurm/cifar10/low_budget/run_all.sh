#!/bin/bash
# Run experiments with resnet18
sbatch resnet18_badge.sh
sbatch resnet18_coreset.sh
sbatch resnet18_random.sh
sbatch resnet18_uncertainty.sh

# Run experiments with resnet18 and label smoothing
# sbatch resnet18-labelsmoothing_random.sh
# sbatch resnet18-labelsmoothing_uncertainty.sh
# sbatch resnet18-labelsmoothing_coreset.sh
# sbatch resnet18-labelsmoothing_badge.sh

# Run experiments with wideresnet2810
# sbatch wideresnet2810_badge.sh
# sbatch wideresnet2810_coreset.sh
# sbatch wideresnet2810_random.sh
# sbatch wideresnet2810_uncertainty.sh
