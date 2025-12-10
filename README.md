# DAL-Toolbox: A PyTorch Toolbox for Deep Active Learning Research
This toolbox is a modular framework designed to facilitate the implementation and evaluation of active learning (AL) workflows in PyTorch.  It includes implementations for the following publications:

| Paper Title | Venue | Year | Code |
| :--- | :---: | :---: | :---: |
| [ActiveGLAE: A Benchmark for Deep Active Learning with Transformers](https://arxiv.org/pdf/2306.10087) | ECML-PKDD | 2023 | [📂 `./publications/aglae`](./publications/aglae) |
| [Role of Hyperparameters in Deep Active Learning](https://ceur-ws.org/Vol-3470/paper4.pdf) | IAL @ ECML-PKDD | 2023 | [📂 `./publications/hyperparameters_in_al`](./publications/hyperparameters_in_al) |
| [Fast Fishing: Approximating Bait for Efficient and Scalable Deep Active Image Classification](https://arxiv.org/pdf/2404.08981) | ECML-PKDD | 2024 | [📂 `./publications/bait_approx`](./publications/bait_approx) |
| [The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification](https://openreview.net/pdf?id=KLBD13bsVl) | TMLR | 2024 | [📂 `./publications/udal`](./publications/udal) |
| [Efficient Bayesian Updates for Deep Active Learning via Laplace Approximations](https://openreview.net/pdf?id=pNSJdyXZju) | ECML-PKDD | 2025 | [📂 `./publications/laplace_updates`](./publications/laplace_updates) |
| [TBD](tbd) | Under Review | 2025 | [📂 `./publications/perf_dal`](./publications/perf_dal) |
| [Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](https://www.arxiv.org/pdf/2511.22344) | Under Review | 2025 | [📂 `./publications/adaptive_al`](./publications/adaptive_al) |

## Getting Started
### Installation 
Setting up the **DAL-Toolbox** is straightforward. Clone the repository and execute the following commands:
```bash
conda create -n dal-toolbox python=3.12
pip install -e .
```
Afterward, install additional packages as required for your task. The implementations in the publication directory typically require additional dependencies, which are aggregated into different `requirements.txt` files.

### Usage Example
The following snippet demonstrates a basic AL cycle on a two-dimensional toy dataset:
```python
import torch
import lightning as L
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import LeastConfidentSampling
from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.deterministic.simplenet import SimpleNet

# 1. Create the two moons dataset
X, y = make_moons(n_samples=200, noise=.1, random_state=42)
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())

# 2. Setup the AL Data Module with 2 initial randomly labeled samples
al_datamodule = ActiveLearningDataModule(dataset, train_batch_size=32)
al_datamodule.random_init(n_samples=2, class_balanced=True)

# 3. Initialize the Model and Strategy
strategy = LeastConfidentSampling()
model = SimpleNet(dropout_rate=0., num_classes=2)
model = DeterministicModel(
    model, 
    optimizer=torch.optim.SGD(model.parameters(), lr=1e-1, momentum=.9)
)

# 4. Perform Active Learning Cycles
for cycle in range(4):
    # Query and update annotations (skip for the initial cycle)
    if cycle != 0:
        indices = strategy.query(model=model, al_datamodule=al_datamodule, acq_size=2)
        al_datamodule.update_annotations(indices)

    # Train the model
    model.reset_states()
    trainer = L.Trainer(max_epochs=50, enable_progress_bar=False)
    trainer.fit(model, al_datamodule)
```

**Note:** While this example uses PyTorch Lightning for convenience, it is not strictly required for most strategies. You can easily replace the L.Trainer with a standard PyTorch training function.

### More Complex Examples

Check out [tbd](https://www.google.com/search?q=tbd) and the [📂`./publications`](./publications) directory for more sophisticated implementations.

## Citation
If you find the DAL-Toolbox useful for your research, you can cite us using:
```
@article{huseljicinterplay,
  title={The Interplay of Uncertainty Modeling and Deep Active Learning: An Empirical Analysis in Image Classification},
  author={Huseljic, Denis and Herde, Marek and Nagel, Yannick and Rauch, Lukas and Strimaitis, Paulius and Sick, Bernhard},
  journal={Transactions on Machine Learning Research}
}
```