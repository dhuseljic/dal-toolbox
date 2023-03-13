#!/bin/bash

# Normal Experiments
# sbatch resnet18_random.sh
# sbatch resnet18_uncertainty.sh
# sbatch resnet18_badge.sh
# sbatch resnet18_coreset.sh
# 
# # Label Smoothing Experiments
sbatch resnet18-labelsmoothing_random.sh
sbatch resnet18-labelsmoothing_uncertainty.sh
# sbatch resnet18-labelsmoothing_badge.sh
# sbatch resnet18-labelsmoothing_coreset.sh
