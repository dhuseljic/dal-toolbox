<a href="https://openreview.net/forum?id=qTs6spvhOS"><img alt="BoSS @ TMLR 2026" src="https://img.shields.io/badge/Paper-BoSS @ TMLR 2026-purple"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2603.13109-b31b1b.svg)](https://arxiv.org/abs/2603.13109)
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning
Here, we provide the implementation and configuration files for reproducing the experiments from our paper [BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](https://openreview.net/forum?id=qTs6spvhOS).

A ready-to-use implementation of BoSS is also available directly in the toolbox under [`dal_toolbox/active_learning/oracles`](../../dal_toolbox/active_learning/oracles/).

## 1. Setup
Ensure `dal-toolbox` is installed. Additionally, install the dependencies required for these experiments:
```bash
pip install hydra-core mlflow rich sentence-transformers
```

## 2. Project Structure
  * **`al.py`**: The main script for running AL experiments with BoSS and baseline strategies.
  * **`oracle.py`**: Implementation of the oracle strategies.
  * **`decision_flips.py`**: Script for the decision-flip experiments.
  * **`configs/`**: YAML configuration files used by hydra.
  * **`slurm/`**: Shell scripts for submitting jobs to a Slurm cluster (includes oracle and strategy comparisons on image and text datasets).
  * **`notebooks/`**: Jupyter notebooks for analyzing results and generating plots.
  * **`utils.py`**: Helper functions.

## 3. Running Experiments

### Local Execution
To run a single experiment locally, execute `al.py`. Ensure you specify the necessary configuration arguments:
```bash
python al.py al.strategy=perf_dal_oracle dataset_name=cifar10
```
### Slurm Cluster Execution
For large-scale reproducibility, use the scripts provided in the `slurm/` directory.

## Citation
```
@article{huseljic2026boss,
  title={BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning},
  author={Huseljic, Denis and Hahn, Paul and Herde, Marek and Sandrock, Christoph and Sick, Bernhard},
  journal={TMLR},
  year={2026}
}
```
