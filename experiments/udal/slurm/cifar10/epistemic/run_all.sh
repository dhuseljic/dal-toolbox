#!/bin/bash

# sbatch resnet18-ensemble_random.sh
# sbatch resnet18-ensemble_entropy.sh
# sbatch resnet18-ensemble_bald.sh
# sbatch resnet18-ensemble_varratio.sh
# 
# sbatch resnet18-mcdropout_random.sh
# sbatch resnet18-mcdropout_entropy.sh
# sbatch resnet18-mcdropout_bald.sh
# sbatch resnet18-mcdropout_varratio.sh

# sbatch resnet18-sngp_random.sh
sbatch resnet18-sngp_entropy.sh
sbatch resnet18-sngp_bald.sh
sbatch resnet18-sngp_varratio.sh
