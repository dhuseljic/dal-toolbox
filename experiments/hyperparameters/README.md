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

|    2K     | HP 1 |      | HP 2 |      | HP 3 |      | HP 4 |      |
|-----------|------|------|------|------|------|------|------|------|
|           | AL   | BO   | AL   | BO   | AL   | BO   | AL   | BO   |
| Random    | todo | todo | todo | todo | todo | todo | todo | todo |
| Entropy   | todo | todo | todo | todo | todo | todo | todo | todo |
| Core-Sets | todo | todo | todo | todo | todo | todo | todo | todo |
| Badge     | todo | todo | todo | todo | todo | todo | todo | todo |

|    4K     | HP 1 |      | HP 2 |      | HP 3 |      | HP 4 |      |
|-----------|------|------|------|------|------|------|------|------|
|           | AL   | BO   | AL   | BO   | AL   | BO   | AL   | BO   |
| Random    | todo | todo | todo | todo | todo | todo | todo | todo |
| Entropy   | todo | todo | todo | todo | todo | todo | todo | todo |
| Core-Sets | todo | todo | todo | todo | todo | todo | todo | todo |
| Badge     | todo | todo | todo | todo | todo | todo | todo | todo |


### CIFAR-100

|    2K     | HP 1 |      | HP 2 |      | HP 3 |      | HP 4 |      |
|-----------|------|------|------|------|------|------|------|------|
|           | AL   | BO   | AL   | BO   | AL   | BO   | AL   | BO   |
| Random    | todo | todo | todo | todo | todo | todo | todo | todo |
| Entropy   | todo | todo | todo | todo | todo | todo | todo | todo |
| Core-Sets | todo | todo | todo | todo | todo | todo | todo | todo |
| Badge     | todo | todo | todo | todo | todo | todo | todo | todo |

|    4K     | HP 1 |      | HP 2 |      | HP 3 |      | HP 4 |      |
|-----------|------|------|------|------|------|------|------|------|
|           | AL   | BO   | AL   | BO   | AL   | BO   | AL   | BO   |
| Random    | todo | todo | todo | todo | todo | todo | todo | todo |
| Entropy   | todo | todo | todo | todo | todo | todo | todo | todo |
| Core-Sets | todo | todo | todo | todo | todo | todo | todo | todo |
| Badge     | todo | todo | todo | todo | todo | todo | todo | todo |