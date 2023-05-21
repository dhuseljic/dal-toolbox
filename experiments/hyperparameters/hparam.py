# Hyperparameter optimization on final datasets obtained through DAL
import json

import lightning as L
import torch
import hydra
import ray
import ray.tune as tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import Subset, random_split, DataLoader
from dal_toolbox import datasets
from dal_toolbox import models
from dal_toolbox import metrics
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger


def train(config, args, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.predict_batch_size, shuffle=False)

    model = models.deterministic.resnet.ResNet18(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    model = models.deterministic.DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy(), 'val_nll': metrics.CrossEntropy()},
    )

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=args.num_epochs,
        callbacks=[MetricLogger(use_print=True)],
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    res = {name: metric.item() for name, metric in trainer.logged_metrics.items() 
           if isinstance(metric, torch.Tensor) and 'val' in name}
    return res


@hydra.main(version_base=None, config_path="./configs", config_name="hparam")
def main(args):
    seed_everything(args.random_seed)

    # Load data
    data = datasets.cifar.CIFAR10('/mnt/datasets')
    dataset = data.train_dataset
    with open(args.queried_indices_json, 'r') as f:
        queried_indices = json.load(f)
    indices = [idx for key in queried_indices for idx in queried_indices[key]]
    dataset = Subset(dataset, indices=indices)

    num_samples = len(dataset)
    num_samples_val = int(args.val_split * len(dataset))
    train_ds, val_ds = random_split(dataset, lengths=[num_samples - num_samples_val, num_samples_val])

    # Start hyperparameter search
    ray.init()
    search_space = {"lr": tune.uniform(1e-4, .5), "weight_decay": tune.uniform(0, .1)}

    objective = tune.with_resources(train, resources={'cpu': args.num_cpus, 'gpu': args.num_gpus})
    objective = tune.with_parameters(objective, args=args, train_ds=train_ds, val_ds=val_ds)

    search_alg = OptunaSearch(points_to_evaluate=[{'lr': 0.001, 'weight_decay': 0.05}])
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.num_opt_samples, metric="val_nll", mode="min")

    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print('Best Hyperparameters: {}'.format(results.get_best_result(metric='val_nll', mode='min').config))


if __name__ == '__main__':
    main()
