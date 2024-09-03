<img src="./icon.png" width="300"/>

# DAL-Toolbox: A Toolbox for Deep Active Learning
Welcome to DAL-Toolbox, a comprehensive repository designed for implementing various models and strategies in deep active learning (DAL). DAL has garnered significant attention for its potential to reduce the amount of labeled data required to train deep neural networks, the most prominent machine learning model of today. Our toolbox provides a versatile and user-friendly framework for researchers and practitioners to explore and advance the field of DAL. Next to classical supervised DAL, we provide tools regarding semi- and self-supervised learning due to their recent success in improving a model's performance as well as tools regarding improving uncertainty as it plays a central role in querying new data in DAL.

## Setup
Setting up the DAL-Toolbox is straight forward! After cloning the repository, use the following comments to get started:
```
conda create -n dal-toolbox python=3.9
pip install -e .
```

## Code Snippet Illustration
The following code snipped demonstrates a basic usage of the DAL-Toolbox on a two-dimensional toy example:

```python
import torch
import torch.nn as nn
import lightning as L
from sklearn.datasets import make_moons
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.active_learning.strategies import LeastConfidentSampling
from dal_toolbox.models.deterministic.simplenet import SimpleNet as Net

# Create the twoo moons dataset
X, y = make_moons(200, noise=.1, random_state=42)

# Transform into a TensorDataset
X, y = torch.tensor(X).float(), torch.tensor(y).long()
tensor_dataset = torch.utils.data.TensorDataset(X, y)

# Setup the AL-Datamodule provided by the dal_toolbox and initialize with two randomly labeled samples
al_datamodule = ActiveLearningDataModule(tensor_dataset, train_batch_size=32)
al_datamodule.random_init(n_samples=2, class_balanced=True)

# Initialize a model and wrap it with the DeterministicModel Wrapper provided by the DAL-Toolbox
model = Net(dropout_rate=0., num_classes=2)
model = DeterministicModel(model, optimizer=torch.optim.SGD(model.parameters(), lr=1e-1, momentum=.9))

# Initialize an AL-Strategy
al_strategy = LeastConfidentSampling()

# Perfom AL-Cycles
for i_cycle in range(8):
    # Acquire new Labels
    if i_cycle != 0:
        indices = al_strategy.query(model=model, al_datamodule=al_datamodule, acq_size=1)
        al_datamodule.update_annotations(indices)

    # Refit the model on the labeled data
    model.reset_states()
    trainer = L.Trainer(max_epochs=50, enable_progress_bar=False)
    trainer.fit(model, al_datamodule)
```

The resulting decision boundary of the model looks as follows

<img src="./examples/readme_example_decision_bounday_1.png" width="500"/>

and shows that a simple deterministic model may not have the best uncertainty estimations to provide good features for LeastCertaintySampling. Let's make use of the implemented Spectral Normalized Gaussian Processes (SNGP) to improve the models uncertainty estimations and hopefully solve this task!

```python
import torch
import torch.nn as nn
import lightning as L
from sklearn.datasets import make_moons
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.models.sngp import SNGPModel
from dal_toolbox.models.deterministic.simplenet import SimpleSNGP as SNGPNet
from dal_toolbox.active_learning.strategies import LeastConfidentSampling

# Create the twoo moons dataset
X, y = make_moons(200, noise=.1, random_state=42)

# Transform into a TensorDataset
X, y = torch.tensor(X).float(), torch.tensor(y).long()
tensor_dataset = torch.utils.data.TensorDataset(X, y)

# Setup the AL-Datamodule provided by the dal_toolbox and initialize with two randomly labeled samples
al_datamodule = ActiveLearningDataModule(tensor_dataset, train_batch_size=32)
al_datamodule.random_init(n_samples=2, class_balanced=True)

# Initialize a model and wrap it with the SNGPModel Wrapper provided by the DAL-Toolbox
model = SNGPNet(num_classes=2, use_spectral_norm=True, spectral_norm_params=spectral_norm_params, gp_params=gp_params)
model = SNGPModel(model, optimizer=torch.optim.SGD(model.parameters(),  lr=1e-2, weight_decay=1e-2, momentum=.9))

# Initialize an AL-Strategy
al_strategy = LeastConfidentSampling()

# Perfom AL-Cycles
for i_cycle in range(8):
    # Acquire new Labels
    if i_cycle != 0:
        indices = al_strategy.query(model=model, al_datamodule=al_datamodule, acq_size=1)
        al_datamodule.update_annotations(indices)

    # Refit the model on the labeled data
    model.reset_states()
    trainer = L.Trainer(max_epochs=50, enable_progress_bar=False)
    trainer.fit(model, al_datamodule)
```

The resulting decision boundary

<img src="./examples/readme_example_decision_bounday_2.png" width="500"/>

looks much more promising, demonstrating how improving the uncertainty estimation of a model can have a positive impact on DAL.

Check out [this notebook](/examples/readme_example.ipynb) which contains the code above with some utility functions to produce the decision boundary plots, ready to be adapted to different scenarios!



## Examples
Next to the example above, we provide various examples for each topic mentioned in the description in the [example section](examples/). Each example folder contains two subfolders, a __toy_example__-folder and a __server_experiments__-folder. The toy examples demonstrate each method on a two dimensional dataset and provide a minimal example how to use the respective methods provided by the toolbox. The server experiments contain an examplatory workflow for working on a cluster server with the DAL-Toolbox. Below, we list each example section provided:

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
The DAL-Toolbox has already been used for various publications. The respective code for their experiments is stored in the __publications__ folder. This may provide relevant insights for experienced researches and plentiful examples of experimental sections in papers with the respective code. Publications using the DAL-Toolbox are

[[1](publications/hyperparameters_in_al/)] Huseljic, Denis, et al. "Role of Hyperparameters in Deep Active Learning." IAL@PKDD/ECML. 2023.

[[2](publications/bait_approx/)] Huseljic, Denis, et al. "Fast Fishing: Approximating BAIT for Efficient and Scalable Deep Active Image Classification." arXiv preprint arXiv:2404.08981 (2024).

[[3](publications/laplace_updates/)] Herde, Marek, et al. "Fast Bayesian Updates for Deep Learning with a Use Case in Active Learning." arXiv preprint arXiv:2210.06112 (2022).

[[4](publications/aglae)] Rauch, Lukas, et al. "Activeglae: A benchmark for deep active learning with transformers." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer Nature Switzerland, 2023.

[[5](publications/udal/)] Huseljic, Denis, et al. "The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification." Transactions on Machine Learning Research.


## Citation
If you find the DAL-Toolbox useful for your research, consider citing us using the following BibTex-citation

```
@article{huseljicinterplay,
  title={The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification},
  author={Huseljic, Denis and Herde, Marek and Nagel, Yannick and Rauch, Lukas and Strimaitis, Paulius and Sick, Bernhard},
  journal={Transactions on Machine Learning Research}
}
```