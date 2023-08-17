import sys
import os
import hydra
import logging

import ray
import ray.tune as tune
import torch
import lightning as L

from omegaconf import OmegaConf
from ray.tune.search.optuna import OptunaSearch
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger


def train(config, args, import_path):
    seed_everything(args.random_seed)
    sys.path.append(import_path)
    from active_learning import build_model, build_datasets, evaluate

    args.model.optimizer.lr = float(config['lr'])
    args.model.optimizer.weight_decay = float(config['weight_decay'])
    if 'mixup_alpha' in config.keys():
        args.model.mixup_alpha = float(config['mixup_alpha'])
    elif 'label_smoothing' in config.keys():
        args.model.label_smoothing = float(config['label_smoothing'])
    elif 'dropout_rate' in config.keys():
        args.model.dropout_rate = float(config['dropout_rate'])
    print("Using model args: {}".format(OmegaConf.to_yaml(args.model)))

    data = build_datasets(args)
    train_ds = data.train_dataset
    val_ds = data.val_dataset

    train_indices = torch.randperm(len(train_ds))[:args.budget]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.model.train_batch_size, sampler=train_indices)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.model.predict_batch_size)

    model = build_model(args, num_classes=data.num_classes)
    trainer = L.Trainer(
        max_epochs=args.model.n_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=args.output_dir,
        callbacks=[MetricLogger(use_print=True)],
        check_val_every_n_epoch=50,
        fast_dev_run=args.fast_dev_run
    )
    trainer.fit(model, train_loader, val_loader)

    predictions = trainer.predict(model, val_loader)
    logits = torch.cat([pred[0] for pred in predictions])
    targets = torch.cat([pred[1] for pred in predictions])
    val_stats = evaluate(logits, targets)
    val_stats = {f'val_{k}': v for k, v in val_stats.items()}

    return val_stats


@hydra.main(version_base=None, config_path="./configs", config_name="hparam_search")
def main(args):
    logger = logging.getLogger()
    logger.info('Using setup: %s', args)

    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', args.cpus_per_trial))
    num_gpus = torch.cuda.device_count()
    ray.init(address='local', num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    # Setup Search space
    search_space, points_to_evaluate = build_search_space(args)
    search_alg = OptunaSearch(points_to_evaluate=points_to_evaluate, seed=args.random_seed)
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=args.n_opt_samples, metric="val_nll", mode="min")

    # Setup objective
    cpus_per_trial = num_cpus // (num_gpus / args.gpus_per_trial)
    objective = tune.with_resources(train, resources={'cpu': cpus_per_trial, 'gpu': args.gpus_per_trial})
    objective = tune.with_parameters(objective, args=args, import_path=os.path.abspath(os.path.dirname(__file__)))

    # Start hyperparameter search
    tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print('Best NLL Stats: {}'.format(results.get_best_result().metrics))
    print('Best NLL Hyperparameter: {}'.format(results.get_best_result().config))
    # print('Best Acc Hyperparameter: {}'.format(results.get_best_result(metric="test_acc1", mode="max").config))


def build_search_space(args):
    points_to_evaluate = None
    if args.model.name == 'resnet18_deterministic':
        search_space = {
            "lr": tune.loguniform(1e-5, .1),
            "weight_decay": tune.loguniform(1e-5, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-2, "weight_decay": 5e-3},
            {"lr": 1e-2, "weight_decay": 5e-2},
        ]
    elif args.model.name == 'resnet18_labelsmoothing':
        search_space = {
            "lr": tune.loguniform(1e-5, .1),
            "weight_decay": tune.loguniform(1e-5, .1),
            "label_smoothing": tune.loguniform(1e-5, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-2, "weight_decay": 5e-3, 'label_smoothing': 0.05},
            {"lr": 1e-2, "weight_decay": 5e-2, 'label_smoothing': 0.05},
        ]
    elif args.model.name == 'resnet18_mixup':
        search_space = {
            "lr": tune.loguniform(1e-5, .1),
            "weight_decay": tune.loguniform(1e-5, .1),
            "mixup_alpha": tune.uniform(0, .4),
        }
        points_to_evaluate = [
            {"lr": 1e-2, "weight_decay": 5e-3, 'mixup_alpha': 0.4},
            {"lr": 1e-2, "weight_decay": 5e-2, 'mixup_alpha': 0.4},
        ]
    elif args.model.name == 'resnet18_mcdropout':
        search_space = {
            "lr": tune.loguniform(1e-5, .1),
            "weight_decay": tune.loguniform(1e-5, .1),
            "dropout_rate": tune.uniform(1e-4, .5),
        }
        points_to_evaluate = [
            {"lr": 1e-2, "weight_decay": 5e-3, 'dropout_rate': 0.1},
            {"lr": 1e-2, "weight_decay": 5e-3, 'dropout_rate': 0.3},
            {"lr": 1e-2, "weight_decay": 5e-2, 'dropout_rate': 0.1},
            {"lr": 1e-2, "weight_decay": 5e-2, 'dropout_rate': 0.3},
        ]
    elif args.model.name == 'resnet18_ensemble':
        # We only optimize a single value for all members
        search_space = {
            "lr": tune.loguniform(1e-5, .1),
            "weight_decay": tune.loguniform(1e-5, .1),
        }
        points_to_evaluate = [
            {"lr": 1e-2, "weight_decay": 5e-3},
            {"lr": 1e-2, "weight_decay": 5e-2},
        ]
    else:
        raise NotImplementedError('Model {} not implemented.'.format(args.model.name))
    return search_space, points_to_evaluate


if __name__ == '__main__':
    main()
