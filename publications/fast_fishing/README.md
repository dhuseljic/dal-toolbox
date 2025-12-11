<a href="https://link.springer.com/chapter/10.1007/978-3-031-70368-3_17"><img alt="fast fishing @ ECML--PKDD 2024" src="https://img.shields.io/badge/Paper-fast fishing @ ECML--PKDD 2024-purple"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2404.08981-b31b1b.svg)](https://arxiv.org/abs/2404.08981)
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

# Fast Fishing: Approximating Bait for Efficient and Scalable Deep Active Image Classification
Here, we provide the implementation and configuration files for reproducing the experiments from our paper [Fast Fishing: Approximating Bait for Efficient and Scalable Deep Active Image Classification](https://arxiv.org/abs/2404.08981).

## 1. Setup
Ensure `dal-toolbox` is installed. Additionally, install the dependencies required for these experiments:
```bash
pip install -r requirements.txt
````
## 2\. Project Structure
  * **`active_learning.py`**: The main script for running AL experiments.
  * **`active_learning.yaml`**: YAML configuration files used by hydra.
  * **`evaluate_active_learning.ipynb`**: Jupyter notebooks for analyzing results and generating plots.
  * **`utils.py`**: Helper functions.

## 3\. Running Experiments

### Local Execution
To run a single experiment locally, execute `main.py`. Ensure you specify the necessary configuration arguments:
```bash
python active_learning.py
```

## Citation
```
@inproceedings{huseljic2024fast,
  title={Fast fishing: Approximating bait for efficient and scalable deep active image classification},
  author={Huseljic, Denis and Hahn, Paul and Herde, Marek and Rauch, Lukas and Sick, Bernhard},
  booktitle={ECML-PKDD},
  year={2024},
}
```