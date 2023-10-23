# Deep Uncertainty Modeling and Active Learning
Framework for uncertainty-based neural networks and active learning.

## Setup
```
conda create -n dal-toolbox python=3.9
pip install -e .
```
### Development
To start developing it is best to use:
```
conda create -n dal-toolbox python=3.9
pip install -e .
```
## Experiments
All experiments are stored in the `experiments` folder.

## Getting Started
We provide [notebooks](notebooks) that give examples of how to work with this repository. 

### Uncertainty modeling
Examples of how to train models with improved uncertainty estimation:
- [Deterministic](notebooks/uncertainty/deterministic.ipynb)
- [Ensemble](notebooks/uncertainty/ensemble.ipynb)
- [MC-Dropout](notebooks/uncertainty/mc-dropout.ipynb)
- [SNGP](notebooks/uncertainty/sngp.ipynb)

### Semi-Supervised Learning
Examples of how to train models with semi-supervised learning algorithms:
- [Fully-Supervised](notebooks/semi_supervised_learning/fully_supervised.ipynb)
- [Pseudo-Labeling](notebooks/semi_supervised_learning/pseudo_labels.ipynb)
- [Pi-Model](notebooks/semi_supervised_learning/pimodel.ipynb)
- [FixMatch](notebooks/semi_supervised_learning/fixmatch.ipynb)

### Active Learning
Examples of how to implement an active learning cycle:
- [Standard Active Learning](notebooks/active_learning/deterministic.ipynb)
- [Bayesian Active Learning](notebooks/active_learning/mc-dropout.ipynb)