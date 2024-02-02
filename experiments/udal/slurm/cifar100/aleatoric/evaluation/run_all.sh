#!/bin/bash
sbatch resnet18-labelsmoothing_random.sh
sbatch resnet18-labelsmoothing_least-confident.sh
sbatch resnet18-labelsmoothing_margin.sh
sbatch resnet18-labelsmoothing_entropy.sh

sbatch resnet18-mixup_random.sh
sbatch resnet18-mixup_least-confident.sh
sbatch resnet18-mixup_margin.sh
sbatch resnet18-mixup_entropy.sh
