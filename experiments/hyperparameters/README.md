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
| Random    | 74.07 ± 0.66 | 77.42 ± 0.29 | 69.69 ± 0.24 | 76.73 ± 0.63 | 69.35 ± 0.78 | 76.52 ± 0.93 | 71.61 ± 0.61 | 77.70 ± 0.57 |
| Entropy   | 74.54 ± 0.31 | 77.63 ± 0.34 | 69.12 ± 1.77 | 77.78 ± 0.76 | 26.08 ± 0.21 | 77.12 ± 0.24 | 16.61 ± 9.34 | 76.61 ± 1.17 |
| Core-Sets | 75.86 ± 0.51 | 77.42 ± 0.09 | 68.75 ± 1.88 | 76.47 ± 0.83 | 43.57 ± 7.43 | 76.76 ± 1.83 | 68.89 ± 1.77 | 76.98 ± 0.88 |
| Badge     | 76.14 ± 0.61 | 78.56 ± 0.44 | 71.47 ± 1.58 | 78.91 ± 0.48 | 59.79 ± 4.74 | 77.61 ± 0.61 | 66.69 ± 3.88 | 78.27 ± 1.37 |

|    4K     | HP 1     |          | HP 2     |         | HP 3     |          | HP 4     |          |
|-----------|----------|----------|----------|---------|----------|----------|----------|----------|
|           | **AL**   | **BO**   | **AL**   | **BO**  | **AL**   | **BO**   | **AL**   | **BO**   |
| Random    | 83.61 ± 0.44 | 83.62 ± 0.36 | 79.91 ± 0.11 | 83.29 ± 0.15 | 75.65 ± 0.85 | 83.38 ± 0.84 | 77.04 ± 0.58 | 83.43 ± 0.68 |
| Entropy   | 85.46 ± 0.29 | 84.07 ± 0.88 | 81.69 ± 0.27 | 84.57 ± 0.86 | 47.23 ± 11.23| 84.02 ± 0.61 | 58.57 ± 17.07| 84.05 ± 0.92 |
| Core-Sets | 85.42 ± 0.62 | 84.98 ± 0.37 | 81.47 ± 0.78 | 84.46 ± 0.90 | 47.98 ± 7.70 | 83.42 ± 0.77 | 71.58 ± 1.58 | 84.30 ± 0.18 |
| Badge     | 85.72 ± 0.17 | 85.12 ± 0.72 | 82.04 ± 0.25 | 84.58 ± 0.29 | 56.81 ± 7.04 | 84.02 ± 0.23 | 71.79 ± 1.57 | 85.50 ± 0.29 |


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