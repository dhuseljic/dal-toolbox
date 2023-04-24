# Deep Uncertainty Modeling and Active Learning

Framework for uncertainty-based neural networks and active learning.

## Setup
```
conda env create -f environment.yml
```

### Development
To start developing it is best to use:
```
conda create -n dal-toolbox python=3.9
pip install -e .
```
## Experiments
All major experiments are stored in the folder `experiments`.

## Notebooks
We provide [notebooks](notebooks) that give examples of how to work with this repository. 

### Uncertainty modeling
Examples of how to realize training with models that improve uncertainty estimation:
- [Deterministic](notebooks/2D-Examples/deterministic.ipynb)
- [Ensemble](notebooks/2D-Examples/ensemble.ipynb)
- [MC-Dropout](notebooks/2D-Examples/mc-dropout.ipynb)
- [SNGP](notebooks/2D-Examples/sngp.ipynb)

### Semi-Supervised Learning
Examples of how to train a model with semi-supervised learning algorithms:
- [Standard Semi-Supervised Learning](notebooks/2D-Examples/semi_supervised_learning.ipynb)

### Active Learning
Examples of how to implement an active learning cycle:
- [Standard Active Learning](notebooks/2D-Examples/active_learning.ipynb)