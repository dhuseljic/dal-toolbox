[![arXiv](https://img.shields.io/badge/arXiv-2511.22344-b31b1b.svg)](https://arxiv.org/abs/2511.22344)
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
# Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning
Here, we provide the implementation and configuration files for reproducing the experiments from our paper [Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](https://www.arxiv.org/pdf/2511.22344).

![Graphical Abstract](/publications/cleaning_the_pool/ga.png)

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
```
@inproceedings{huseljic2026refine,
	title = {Cleaning the {Pool}: {Progressive} {Filtering} of {Unlabeled} {Pools} in {Deep} {Active} {Learning}},
	shorttitle = {Cleaning the {Pool}},
	author = {Huseljic, Denis and Herde, Marek and Rauch, Lukas and Hahn, Paul and Sick, Bernhard},
	booktitle = {CVPR},
	year = {2026},
}
```
