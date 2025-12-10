[![arXiv](https://img.shields.io/badge/arXiv-2511.22344-b31b1b.svg)](https://arxiv.org/abs/2511.22344)
# Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning
This directory contains the implementation and configuration files required to reproduce the experiments presented in the paper [Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](https://www.arxiv.org/pdf/2511.22344).

## 1. Setup & Requirements
Ensure the core `dal-toolbox` is installed. Additionally, install the specific dependencies required for these experiments:
```bash
pip install -r requirements.txt
````

## 2\. Project Structure
  * **`main.py`**: The primary entry point for executing Active Learning (AL) cycles and model training.
  * **`configs/`**: YAML configuration files.
      * `active_learning.yaml`: General AL parameters.
      * `dataset/`: Dataset-specific configurations (e.g., `cifar10.yaml`, `imagenet.yaml`).
  * **`slurm/`**: Shell scripts for submitting jobs to a Slurm cluster (includes ablations, grid searches, and baselines).
  * **`strategies.py`**: Implementation of experiment-specific query strategies.
  * **`utils.py`**: Helper functions for data loading and logging.
  * **`*.ipynb`**: Jupyter notebooks (`evaluate.ipynb`, `tsne.ipynb`) for analyzing results and generating plots.

## 3\. Running Experiments

### Local Execution

To run a single experiment locally, execute `main.py`. Ensure you specify the necessary configuration arguments (depending on your argument parser, e.g., Hydra or argparse):
```bash
# Example usage
python main.py --config configs/active_learning.yaml --dataset configs/dataset/cifar10.yaml
```

### Slurm Cluster Execution
For large-scale reproducibility, use the scripts provided in the `slurm/` directory.
  * **Standard AL Loop:** `sbatch slurm/al.sh`
  * **Ablation Studies:**
      * `slurm/ablation_alpha.sh` (Hyperparameter alpha)
      * `slurm/ablation_batches.sh` (Batch size)
      * `slurm/ablation_depth.sh` (Model depth)
  * **Baselines:** `sbatch slurm/baselines.sh`
```

## Citation
TBD