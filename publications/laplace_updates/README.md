<a href="https://link.springer.com/chapter/10.1007/978-3-032-05981-9_2"><img alt="laplace updates @ ECML-PKDD 2025" src="https://img.shields.io/badge/Paper-laplace updates @ ECML-PKDD 2025-purple"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2210.06112-b31b1b.svg)](https://arxiv.org/abs/2210.06112)
# Efficient Bayesian Updates for Deep Active Learning via Laplace Approximations
Here, we provide the implementation and configuration files for reproducing the experiments from our paper [Efficient Bayesian Updates for Deep Active Learning via Laplace Approximations](https://arxiv.org/pdf/2210.06112).

## 1. Setup
Ensure `dal-toolbox` is installed. Additionally, install the dependencies required for these experiments:
```bash
pip install -r requirements.txt
````
## 2\. Project Structure
  * **`main.py`**: The main script for running AL experiments.
  * **`configs/`**: YAML configuration files used by hydra.
  * **`slurm/`**: Shell scripts for submitting jobs to a Slurm cluster (includes ablations, grid searches, and baselines).
  * **`strategies.py`**: Implementation of ensemble AL methods.
  * **`utils.py`**: Helper functions.
  * **`*.ipynb`**: Jupyter notebooks for analyzing results and generating plots.

## 3\. Running Experiments

### Local Execution
To run a single experiment locally, execute `main.py`. Ensure you specify the necessary configuration arguments:
```bash
python main.py al.strategy=refine dataset=cifar10
```
### Slurm Cluster Execution
For large-scale reproducibility, use the scripts provided in the `slurm/` directory.

## Citation
TBD