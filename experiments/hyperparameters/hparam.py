# Hyperparameter optimization on final datasets obtained through DAL
import os
import json

import lightning as L
import torch
import hydra
import ray
from ray import tune, air

from omegaconf import OmegaConf
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from dal_toolbox import datasets
from dal_toolbox import models
from dal_toolbox import metrics
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup


def train(config, args, al_dataset, num_classes):
    seed_everything(100 + args.random_seed)

    all_val_stats = []
    kf = KFold(n_splits=args.num_folds, shuffle=True)
    indices = range(len(al_dataset))
    for train_indices, val_indices in kf.split(indices):
        train_ds = Subset(al_dataset, indices=train_indices)
        val_ds = Subset(al_dataset, indices=val_indices)

        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.predict_batch_size, shuffle=False)

        # Create model
        model = build_model(args, lr=config['lr'], weight_decay=config['weight_decay'], num_classes=num_classes)

        # Train model
        trainer = L.Trainer(
            enable_checkpointing=False,
            max_epochs=args.num_epochs,
            callbacks=[MetricLogger(use_print=True)],
            enable_progress_bar=False,
            default_root_dir=args.output_dir,
            fast_dev_run=args.fast_dev_run,
            logger=False,
        )
        trainer.fit(model, train_loader, val_dataloaders=val_loader)
        val_stats = trainer.validate(model, val_loader)[0]
        all_val_stats.append(val_stats)
    avg_val_stats = {key: sum(d[key] for d in all_val_stats) / len(all_val_stats) for key in all_val_stats[0]}
    return avg_val_stats


def build_model(args, lr, weight_decay, num_classes):
    model = models.deterministic.resnet.ResNet18(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=.9)
    lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.num_epochs, warmup_epochs=10)
    model = models.deterministic.DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy(), 'val_nll': metrics.CrossEntropy()}
    )
    return model


@hydra.main(version_base=None, config_path="./configs", config_name="hparam")
def main(args):
    print(OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.dataset == 'CIFAR10':
        data = datasets.cifar.CIFAR10(args.dataset_path)
    elif args.dataset == 'CIFAR100':
        data = datasets.cifar.CIFAR100(args.dataset_path)
    else:
        raise NotImplementedError

    with open(args.queried_indices_json, 'r') as f:
        queried_indices = json.load(f)
    if args.budget == 2000:
        indices = [idx for key in queried_indices if key in [
            f'cycle{i}' for i in range(20)] for idx in queried_indices[key]]
    elif args.budget == 4000:
        indices = [idx for key in queried_indices for idx in queried_indices[key]]
    else:
        raise NotImplementedError('Check the budget argument.')
    assert len(indices) == args.budget, 'Something went wrong with the queried indices'

    al_dataset = Subset(data.train_dataset, indices=indices)

    # Start hyperparameter search
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', args.num_cpus))
    num_gpus = torch.cuda.device_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True,
             _temp_dir=os.path.join(args.output_dir, 'tmp'))

    search_space = {"lr": tune.loguniform(1e-5, .1), "weight_decay": tune.loguniform(1e-5, .1)}
    objective = tune.with_resources(train, resources={'cpu': args.num_cpus, 'gpu': args.num_gpus})
    objective = tune.with_parameters(objective, args=args, al_dataset=al_dataset, num_classes=data.num_classes)
    search_alg = OptunaSearch(points_to_evaluate=[{'lr': args.lr, 'weight_decay': args.weight_decay}])
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.num_opt_samples, metric="val_acc", mode="max")
    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config,
                       run_config=air.RunConfig(storage_path=args.output_dir))
    results = tuner.fit()
    print('Best Hyperparameters for NLL: {}'.format(results.get_best_result(metric='val_nll', mode='min').config))
    print('Best Hyperparameters for ACC: {}'.format(results.get_best_result(metric='val_acc', mode='max').config))

    # Fit model with best hyperparameters
    best_config = results.get_best_result(metric='val_acc', mode='max').config
    print(f'Training final model using the best possible parameters {best_config}')
    train_loader_all = DataLoader(al_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    model = build_model(args, lr=best_config['lr'],
                        weight_decay=best_config['weight_decay'], num_classes=data.num_classes)
    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=args.num_epochs,
        callbacks=[MetricLogger(use_print=True)],
        enable_progress_bar=False,
        devices=1,
    )
    trainer.fit(model, train_loader_all)

    test_loader = DataLoader(data.test_dataset, batch_size=args.predict_batch_size, shuffle=False)
    predictions_id = trainer.predict(model, test_loader)
    logits_id = torch.cat([pred[0] for pred in predictions_id])
    targets_id = torch.cat([pred[1] for pred in predictions_id])
    test_stats = {
        'test_acc': metrics.Accuracy()(logits_id, targets_id).item(),
        'test_nll': metrics.CrossEntropy()(logits_id, targets_id).item()
    }
    print(f'Final results: {test_stats}')

    result_list = []
    for res in results:
        result_list.append({'res': res.metrics, 'conf': res.config})

    print('Saving results.')
    history = {
        'args': OmegaConf.to_yaml(args),
        'test_stats': test_stats,
        'ray-results': result_list,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    main()
