<img src="./icon.png" width="200"/>

Welcome to DAL-Toolbox, a comprehensive repository designed for implementing various models and strategies in deep active learning (DAL). DAL has garnered significant attention for its potential to reduce the amount of labeled data required to train deep neural networks, the most prominent machine learning model of today. Our toolbox provides a versatile and user-friendly framework for researchers and practitioners to explore and advance the field of DAL. Next to classical supervised DAL, we provide tools regarding semi- and self-supervised learning due to their recent success as well as improving uncertainty as it plays a central role in DAL.

## Setup

Setting up the DAL-Toolbox is straight forward! After cloning the repository, use the following comments to get started:
```
conda create -n dal-toolbox python=3.9
pip install -e .
```

## Examples

We provide various examples for each topic mentioned in the description in the [example section](examples/). Each example folder contains two subfolders, a __toy_example__-folder and a __server_experiments__-folder. The toy examples demonstrate each method on a two dimensional dataset and provide a minimal example how to use the respective methods provided by the toolbox. The server experiments contain an examplatory workflow for working on a cluster server with the DAL-Toolbox. Below, we list each example section provided:

### Active Learning
Examples of how to implement an active learning cycle:
- [Standard Active Learning](examples/active_learning/toy_examples/deterministic.ipynb)
- [Bayesian Active Learning](examples/active_learning/toy_examples/mc-dropout.ipynb)

### Self-Supervised Learning
Examples of how to train models with self-supervised learning algorithms:
- [SimCLR](examples/self_supervised_learning/toy_examples/simclr.ipynb)

### Semi-Supervised Learning
Examples of how to train models with semi-supervised learning algorithms:
- [Pseudo-Labeling](examples/semi_supervised_learning/toy_examples/pseudo_labels.ipynb)
- [Pi-Model](examples/semi_supervised_learning/toy_examples/pimodel.ipynb)
- [FixMatch](examples/semi_supervised_learning/toy_examples/fixmatch.ipynb)

### Uncertainty
Examples of how to train models with improved uncertainty estimation:
- [Deterministic](examples/uncertainty/toy_examples/deterministic.ipynb)
- [Ensemble](examples/uncertainty/toy_examples/ensemble.ipynb)
- [MC-Dropout](examples/uncertainty/toy_examples/mc-dropout.ipynb)
- [SNGP](examples/uncertainty/toy_examples/sngp.ipynb)


## Publications

The DAL-Toolbox has already been used for various publications. The respective code for their experiments is stored in the __publication_experiments__ folder. This may provide relevant insights for experienced researches and plentiful examples of experimental sections in papers with the respective code. Publications using the DAL-Toolbox are
 
TODO: Insert Publication References


## Citation

TODO: Should we include a citation?