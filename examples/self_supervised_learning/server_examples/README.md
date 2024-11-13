# Server Examples for Self-Supervised Learning
This example section aims to demonstrate how to use the DAL-Toolbox to pretrain a DNN using an auxiliary task to learn meaningful features for downstream tasks.

## Methods for Self-Supervised Learning
Currently, only one method is implemented. SimCLR duplicates a batch of samples while augmenting each batch differently. The subsequent auxiliary task for the model is to find the pairs of samples that belong to the same original but are differently augmented.

## Usage
In general, an experiment can be started by running the command
```
srun python pretrain.py
```

for pretraining a model and 

```
srun python finetune.py
```

For any arguments, we refer to the [config file](./configs/config.yaml) for greater details.