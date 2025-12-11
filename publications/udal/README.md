<a href="https://openreview.net/pdf?id=KLBD13bsVl"><img alt="DAL uncertainty @ TMLR 2024" src="https://img.shields.io/badge/Paper-DAL uncertainty @ TMLR 2024-purple"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

# The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification
Here, we provide the implementation and configuration files for reproducing the experiments from our paper [The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification](https://openreview.net/pdf?id=KLBD13bsVl).

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
@article{huseljic2024interplay,
  title={The interplay of uncertainty modeling and deep active learning: An empirical analysis in image classification},
  author={Huseljic, Denis and Herde, Marek and Nagel, Yannick and Rauch, Lukas and Strimaitis, Paulius and Sick, Bernhard},
  journal={TMLR},
  year={2024}
}
```
