# Hyperparameter optimization on final datasets obtained through DAL
import os
import json
from omegaconf import DictConfig, OmegaConf

import lightning as L
import torch
import hydra
import ray
import ray.tune as tune

from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import Subset, random_split, DataLoader
from dal_toolbox import datasets
from dal_toolbox import models
from dal_toolbox import metrics
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger


def train(config, args, train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.predict_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.predict_batch_size, shuffle=False)

    model = models.deterministic.resnet.ResNet18(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    model = models.deterministic.DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy(), 'val_nll': metrics.CrossEntropy()}
    )

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=args.num_epochs,
        callbacks=[MetricLogger(use_print=True)],
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    logged_metrics = trainer.logged_metrics

    # Evaluation
    predictions_id = trainer.predict(model, test_loader)
    logits_id = torch.cat([pred[0] for pred in predictions_id])
    targets_id = torch.cat([pred[1] for pred in predictions_id])

    test_stats = {
        'test_acc': metrics.Accuracy()(logits_id, targets_id).item(), 
        'test_nll': metrics.CrossEntropy()(logits_id, targets_id).item()
    }
    
    res = {name: metric.item() for name, metric in logged_metrics.items() 
           if isinstance(metric, torch.Tensor)} | test_stats
    return res 


@hydra.main(version_base=None, config_path="./configs", config_name="hparam")
def main(args):
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = datasets.cifar.CIFAR10(args.dataset_path)
    dataset = data.train_dataset
    with open(args.queried_indices_json, 'r') as f:
        queried_indices = json.load(f)
    indices = [idx for key in queried_indices for idx in queried_indices[key]]
    dataset = Subset(dataset, indices=indices)

    num_samples = len(dataset)
    num_samples_val = int(args.val_split * len(dataset))
    train_ds, val_ds = random_split(dataset, lengths=[num_samples - num_samples_val, num_samples_val])
    test_ds = data.test_dataset

    # Start hyperparameter search
    ray.init()
    search_space = {"lr": tune.uniform(1e-4, .5), "weight_decay": tune.uniform(0, .1)}

    objective = tune.with_resources(train, resources={'cpu': args.num_cpus, 'gpu': args.num_gpus})
    objective = tune.with_parameters(objective, args=args, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)

    search_alg = OptunaSearch(points_to_evaluate=[{'lr': 0.001, 'weight_decay': 0.05}])
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.num_opt_samples, metric="val_nll", mode="min")

    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print('Best Hyperparameters for NLL: {}'.format(results.get_best_result(metric='val_nll', mode='min').config))
    print('Best Hyperparameters for ACC: {}'.format(results.get_best_result(metric='val_acc', mode='max').config))

    result_list = []

    for res in results:
        result_list.append({
            'res':res.metrics,
            'conf':res.config
            })

    history = {
        'args':OmegaConf.to_yaml(args),
        'ray-results':result_list
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    main()
