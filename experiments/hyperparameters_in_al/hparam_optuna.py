# Hyperparameter optimization on final datasets obtained through DAL
import os
import json
import hydra
from omegaconf import OmegaConf
import joblib

import lightning as L
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold

from dal_toolbox import datasets
from dal_toolbox import models
from dal_toolbox import metrics
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup

import optuna


def objective(trial, args, al_dataset, num_classes):
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
        model = build_model(args, lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True), weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True), num_classes=num_classes)

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
    return avg_val_stats['val_acc']


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

    #with open(args.queried_indices_json, 'r') as f:
    #    queried_indices = json.load(f)
    #if args.budget == 2000:
    #    indices = [idx for key in queried_indices if key in [
    #        f'cycle{i}' for i in range(20)] for idx in queried_indices[key]]
    #elif args.budget == 4000:
    #    indices = [idx for key in queried_indices for idx in queried_indices[key]]
    #else:
    #    raise NotImplementedError('Check the budget argument.')
    #assert len(indices) == args.budget, 'Something went wrong with the queried indices'
    indices = torch.randint(0, 45000, (2000,))
    al_dataset = Subset(data.train_dataset, indices=indices)

    # Start hyperparameter search
    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    study_name = "Test"
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    print(f"Sampler is {study.sampler.__class__.__name__}")

    # Suggest Trials
    study.enqueue_trial(
        {
            "lr": 1e-2,
            "weight_decay0": 5e-4
        }
    )

    # Run optimization
    study.optimize(lambda trial: objective(trial, args, al_dataset, data.num_classes), n_trials=args.num_opt_samples)

    # Best params
    best_config = study.best_params
    results = study.trials

    print("Results of the optimization")
    print(results)

    # Fit model with best hyperparameters
    print(f'Training final model using the best possible parameters {best_config}')
    train_loader_all = DataLoader(al_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    model = build_model(args, lr=best_config['lr'],
                        weight_decay=best_config['weight_decay'], num_classes=data.num_classes)
    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=args.num_epochs,
        callbacks=[MetricLogger(use_print=True)],
        enable_progress_bar=False,
        fast_dev_run=args.fast_dev_run,
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

    print('Saving results.')
    history = {
        'args': OmegaConf.to_yaml(args),
        'test_stats': test_stats
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)

    with open(os.path.join(args.output_dir, "study.pkl"), 'wb') as g:
        joblib.dump(study, g)


if __name__ == '__main__':
    main()
