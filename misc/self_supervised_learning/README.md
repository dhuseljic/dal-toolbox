# Self Supervised Learning

## Installation

First install all dependencies of `dal-toolbox`, which are located [here](../../requirements.txt).
Then install the requirements specified in [requirements.txt](requirements.txt)

## Training a self supervised model

To train a model simply run: `python main.py`

This will use the standard hyperparameters specified in [configs/config.yaml](config/config.yaml).
You can change these parameters either by adjusting the config file, or passing different parameters to run, e.g. `python main.py ssl_model=YOUR_MODEL`.

### Hyperparameters

| Argument                          | Standard Parameter | Description                                                         |
|-----------------------------------|--------------------|---------------------------------------------------------------------|
| `ssl_model`                       | `simclr`           | The self-supervised model to train. Overview found [here](#models)  |
| `dataset`                         | `CIFAR10`          | The dataset to use. Overview found [here](#datasets)                |
| `le_model.num_epochs`             | `90`               | The number of epoch the linear evaluation model is trained for.     |
| `le_model.train_batch_size`       | `4096`             | The training batch size for the linear evaluation model.            |
| `le_model.val_batch_size`         | `512`              | The validation batch size for the linear evaluation model.          |
| `le_model.optimizer.base_lr`      | `0.1`              | The base learning rate for the linear evaluation model.             |
| `le_model.optimizer.weight_decay` | `0.0`              | The weight decay for the linear evaluation model.                   |
| `le_model.optimizer.momentum`     | `0.9`              | The momentum for the linear evaluation model.                       |
| `le_model.callback.enabled`       | `True`             | Whether the linear evaluation callback is enabled.                  |
| `le_model.callback.interval`      | `50`               | The interval in which the linear evaluation takes place if enabled. |
| `n_cpus`                          | `16`               | The number of CPUs used for the torch dataloader.                   |
| `random_seed`                     | `42`               | The random seed for reproducibility.                                |
| `ssl_val_interval`                | `25`               | Every `ssl_val_interval` epochs the validation step is done.        |
| `dataset_path`                    | `./data/`          | The directory were the datasets are stored/downloaded.              |
| `output_dir`                      | `./output/`        | The directory where the results/logs/checkpoints are saved to.      |

It is also possible to use precomputed features (e.g. from a self-supervised task) for training.
See [Using precomputed features](#using-precomputed-features) for more details.

#### Models

The following self-supervised models are implemented:

| Model                  | Argument |
|------------------------|----------|
| SimCLR [[1](#sources)] | `simclr` |

Furthermore, the following hyperparameters can be adjusted for the SimCLR model:

| Argument                       | Description                                                       |
|--------------------------------|-------------------------------------------------------------------|
| `model.name`                   | The name of the model.                                            |
| `model.encoder`                | Which encoder to use for SimCLR.                                  |
| `model.projector`              | Which projector to use for SimCLR.                                |
| `model.projector_dim`          | The projectors dimension.                                         |
| `model.temperature`            | The temperature for the SimCLR loss.                              |
| `model.n_epochs`               | How many epochs the model should be trained for in each AL cycle. |
| `model.train_batch_size`       | The batch size for training.                                      |
| `model.accumulate_batches`     | How many batches are accumulated before stepping the optimizer.   |
| `model.predict_batch_size`     | The batch size for validation/testing.                            |
| `model.optimizer.base_lr`      | The base learning rate for training.                              |
| `model.optimizer.weight_decay` | The weight decay for training.                                    |
| `model.optimizer.momentum`     | The momentum for training.                                        |

The standard parameters depend on each specific model and can be found in their respective config file.

#### Datasets

The following datasets are implemented:

| Dataset                     | Argument      |
|-----------------------------|---------------|
| CIFAR10 [[2](#sources)]     | `CIFAR10`     |
| CIFAR100 [[2](#sources)]    | `CIFAR100`    |
| SVHN  [[3](#sources)]       | `SVHN`        |
| ImageNet [[4](#sources)]    | `ImageNet`    |
| ImageNet50 [[5](#sources)]  | `ImageNet50`  |
| ImageNet100 [[5](#sources)] | `ImageNet100` |
| ImageNet200 [[5](#sources)] | `ImageNet200` |

Keep in mind that ImageNet and its subsets are not automatically downloaded and have to be downloaded manually (See https://image-net.org/).
You can also adjust the with strength with which the colors are distorted with `color_distortion_strength=VALUE`.

## Sources

- [1] Chen, Ting, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. “A Simple Framework for Contrastive Learning of Visual Representations.” In Proceedings of the 37th International Conference on Machine Learning, 119:1597–1607. Proceedings of Machine Learning Research. Virtual: PMLR, 2020. https://proceedings.mlr.press/v119/chen20j.html.
- [2] Krizhevsky, Alex. “Learning Multiple Layers of Features from Tiny Images.” (2009).
- [3] Netzer, Yuval, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. “Reading Digits in Natural Images with Unsupervised Feature Learning,.” In Deep Learning Unsupervised Feature Learn. Workshop @ Adv. Neural. Inf. Process. Syst. Granada, Spain, 2011.
- [4] Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. “ImageNet: A Large-Scale Hierarchical Image Database.” In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–55. Miami, USA: IEEE, 2009. https://doi.org/10.1109/CVPR.2009.5206848.
- [5] Van Gansbeke, Wouter, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. “SCAN: Learning to Classify Images without Labels.” In Computer Vision -- ECCV 2020, 12355:268–85. Lecture Notes in Computer Science. Glasgow, UK: Springer International Publishing, 2020. https://doi.org/10.1007/978-3-030-58607-2_16.



