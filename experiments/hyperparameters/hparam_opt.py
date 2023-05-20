# Hyperparameter optimization on final datasets obtained through DAL
import json
import logging

import lightning as L
import torch
import hydra
import ray
import ray.tune as tune

from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import Subset, random_split, DataLoader
from dal_toolbox import datasets
from dal_toolbox import models
from dal_toolbox.utils import seed_everything
from dal_toolbox.metrics import Accuracy
from dal_toolbox.models.utils.callbacks import MetricLogger

queried_indices_json = "/mnt/work/dhuseljic/results/hyperparameters/graphical_abstract/badge/lr0.001_wd0.05/seed1/queried_indices.json"
random_seed = 42
val_split = 0.1
num_epochs = 200
train_batch_size = 32
predict_batch_size = 256


def train(config, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=predict_batch_size, shuffle=False)

    model = models.deterministic.resnet.ResNet18(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model = models.deterministic.DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=num_epochs,
        callbacks=[MetricLogger(use_print=True)],
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader)

    predictions = trainer.predict(model, val_loader)
    logits = torch.cat([preds[0] for preds in predictions])
    targets = torch.cat([preds[1] for preds in predictions])
    acc_fn = Accuracy()

    return {'accuracy': acc_fn(logits, targets).item()}


def main():
    seed_everything(random_seed)

    # Load data
    dataset = datasets.cifar.build_cifar10('train', '/mnt/datasets')
    with open(queried_indices_json, 'r') as f:
        queried_indices = json.load(f)
    indices = [idx for key in queried_indices for idx in queried_indices[key]]
    dataset = Subset(dataset, indices=indices)

    num_samples = len(dataset)
    num_samples_val = int(val_split * len(dataset))
    train_ds, val_ds = random_split(dataset, lengths=[num_samples - num_samples_val, num_samples_val])

    # Start hyperparameter search
    ray.init()
    search_space = {"lr": tune.uniform(1e-4, .5), "weight_decay": tune.uniform(0, .1)}

    objective = tune.with_resources(train, resources={'cpu': 4, 'gpu': 1})
    objective = tune.with_parameters(objective, train_ds=train_ds, val_ds=val_ds)

    search_alg = OptunaSearch(points_to_evaluate=[{'lr': 0.001, 'weight_decay': 0.05}])
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=100, metric="accuracy", mode="max")

    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print('Best Hyperparameters: {}'.format(results.get_best_result(metric='accuracy', mode='max').config))


if __name__ == '__main__':
    main()
