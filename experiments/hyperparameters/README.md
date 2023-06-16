# Study on Hyperparameters in Deep Active Learning
This is the official implementation for the paper *Role of Hyperparameters in Deep Active Learning*.

## Setup
To run the provided python scripts you need to set up and use the following environment.
```
conda create -n dal-toolbox python=3.9
conda activate dal-toolbox
pip install .
pip install -U "ray[tune]"
pip install hydra-core optuna
```
Make sure you are in the root directory of this repository to correctly install the dal-toolbox package.

## Reproducibility
All experiments conducted in the paper are located under the `./slurm` directory.

## Results
Here, we report results for the CIFAR-10 and CIFAR-100 datasets.

### CIFAR-10

|    2K     | HP 1     |          | HP 2     |         | HP 3     |          | HP 4     |          |
|-----------|----------|----------|----------|---------|----------|----------|----------|----------|
|           | **AL**   | **BO**   | **AL**   | **BO**  | **AL**   | **BO**   | **AL**   | **BO**   |
| Random    | 74.07 ± 0.66 | todo | 69.69 ± 0.24 | todo | 69.35 ± 0.78 | todo | 71.61 ± 0.61 | todo     |
| Entropy   | 74.54 ± 0.31 | todo | 69.12 ± 1.77 | todo | 26.08 ± 0.21 | todo | 16.61 ± 9.34 | todo     |
| Core-Sets | 75.86 ± 0.51 | todo | 68.75 ± 1.88 | todo | 43.57 ± 7.43 | todo | 68.89 ± 1.77 | todo     |
| Badge     | 76.14 ± 0.61 | todo | 71.47 ± 1.58 | todo | 59.79 ± 4.74 | todo | 66.69 ± 3.88 | todo     |

|    4K     | HP 1     |          | HP 2     |         | HP 3     |          | HP 4     |          |
|-----------|----------|----------|----------|---------|----------|----------|----------|----------|
|           | **AL**   | **BO**   | **AL**   | **BO**  | **AL**   | **BO**   | **AL**   | **BO**   |
| Random    | 83.61 ± 0.44 | todo     | 79.91 ± 0.11 | todo    | 75.65 ± 0.85 | todo     | 77.04 ± 0.58 | todo     |
| Entropy   | 85.46 ± 0.29 | todo     | 81.69 ± 0.27 | todo    | 47.23 ±11.23 | todo     | 58.57 ±17.07 | todo     |
| Core-Sets | 85.42 ± 0.62 | todo     | 81.47 ± 0.78 | todo    | 47.98 ± 7.70 | todo     | 71.58 ± 1.58 | todo     |
| Badge     | 85.72 ± 0.17 | todo     | 82.04 ± 0.25 | todo    | 56.81 ± 7.04 | todo     | 71.79 ± 1.57 | todo     |


### CIFAR-100

|    2K     | HP 1     |          | HP 2     |         | HP 3     |          | HP 4     |          |
|-----------|----------|----------|----------|---------|----------|----------|----------|----------|
|           | **AL**   | **BO**   | **AL**   | **BO**  | **AL**   | **BO**   | **AL**   | **BO**   |
| Random    | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Entropy   | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Core-Sets | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Badge     | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |

|    4K     | HP 1     |          | HP 2     |         | HP 3     |          | HP 4     |          |
|-----------|----------|----------|----------|---------|----------|----------|----------|----------|
|           | **AL**   | **BO**   | **AL**   | **BO**  | **AL**   | **BO**   | **AL**   | **BO**   |
| Random    | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Entropy   | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Core-Sets | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |
| Badge     | todo     | todo     | todo     | todo    | todo     | todo     | todo     | todo     |