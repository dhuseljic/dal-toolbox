# Active Learning Server Experiment

## Installation

First create a virtual environment with `python >= 3.9`.
Then install the `dal-toolbox` with `pip install -e .` on the highest folder level.
Finally install any additional requirements placed in [requirements.txt](requirements.txt) with `pip install -r requirements.txt`.

## Usage

To train a model simply run: `python main.py`

This will use the standard hyperparameters specified in [configs/config.yaml](configs/active_learning.yaml).
You can change these parameters either by adjusting the config file, or passing different parameters to run, e.g. `python active_learning.py model=YOUR_MODEL`.

## Hyperparameters

| Argument                 | Standard Parameter       | Description                                                                             |
|--------------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `model`                  | `resnet18`                 | The model to train. Overview found [here](#models)                                      |
| `dataset`                | `CIFAR10`                | The dataset to use. Overview found [here](#datasets)                                    |
| `ssl_algorithm`            | `random`                 | The ssl_algorithm to use. Overview found [here](#ssl-algorithms) |
| `random_seed`            | `42`                     | The random seed for reproducibility.                                                    |
| `val_interval`           | `25`                     | Every `val_interval` epochs the validation step is done.                                |
| `data_dir`           | `./data/`                | The directory were the datasets are stored/downloaded.                                  |
| `output_dir`             | `./output/`              | The directory where the results/logs/checkpoints are saved to.                          |
| `num_labeled`             | 40              | The number of labeled samples to use during training.                         |
| `num_unlabeled`             | 50000              | The number of unlabeled samples to use during training.                         |


## Models

The following models are implemented:

| Model                          | Argument                       |
|--------------------------------|--------------------------------|
| ResNet18 [[1](#sources)]       | `resnet18`                     |
| WideResNet282 [[3](#sources)]         | `wideresnet282`         |
| WideResNet2810 [[3](#sources)]         | `wideresnet2810`       |


Furthermore, the following hyperparameters can be adjusted for each model:

| Argument                       | Description                                                       |
|--------------------------------|-------------------------------------------------------------------|
| `model.num_epochs`             | How many epochs the model should be trained for.                  |
| `model.train_batch_size`       | The batch size for training.                                      |
| `model.predict_batch_size`     | The batch size for validation/testing.                            |
| `model.optimizer.lr`           | The learning rate for training.                                   |
| `model.optimizer.weight_decay` | The weight decay for training.                                    |
| `model.optimizer.momentum`     | The momentum for training.                                        |

The standard parameters depend on each specific model and can be found in their respective config file.
(For example, the config for the ResNet18 con be found in [configs/model/resnet18.yaml](configs/model/resnet18.yaml).)


## Datasets

For Semi-SL, only the following datasets are implemented.

| Dataset                     | Argument      |
|-----------------------------|---------------|
| CIFAR10 [[2](#sources)]     | `CIFAR10`     |

## Semi-Supervised-Learning Algorithms

The following active learning strategies are implemented:

| Strategy                  | Argument    |
|---------------------------|-------------|
| PseudoLabels [[4](#sources)]          | `pseudo_labels`    |
| PiModel [[5](#sources)]        | `pi_model`   |
| FixMatch [[6](#sources)]          | `fixmatch` |
| FlexMatch [[7](#sources)] | `flexmatch` |

Furthermore, for each algorithm, the hyperparameters `ssl_algorithm.u_ratio` and `ssl_algorithm.lambda_u` can be adjusted, determining the ratio of labeled and unlabeled samples to see per batch and how to weigh the unlabeled loss in contrast to the labeled loss.

## Baseline Results

Here we see an overview of all baseline experiments performed.
All slurm scripts used to run these experiments can be found [here](slurm/).

TODO: Insert one single plot of Learning Curves concerning Supervised Baseline and each semi-sl algorithm? 

## Sources

- [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. “Deep Residual Learning for Image Recognition.” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–78, 2016.
- [2] Krizhevsky, Alex. “Learning Multiple Layers of Features from Tiny Images.” (2009).
- [3] Zagoruyko, Sergey. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).
- [4] H. Wu and S. Prasad, "Semi-Supervised Deep Learning Using Pseudo Labels for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 27, no. 3, pp. 1259-1270, March 2018
- [5] Laine, Samuli, and Timo Aila. "Temporal Ensembling for Semi-Supervised Learning." International Conference on Learning Representations. 2022.
- [6] Sohn, Kihyuk, et al. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." Advances in neural information processing systems 33 (2020): 596-608.
- [7] Zhang, Bowen, et al. "Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling." Advances in Neural Information Processing Systems 34 (2021): 18408-18419.