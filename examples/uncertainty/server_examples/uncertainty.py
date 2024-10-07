import os
import json
import logging

import hydra
import torch
import torch.nn as nn
from torch import Tensor
import lightning as L

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dal_toolbox import datasets
from dal_toolbox.datasets.base import BaseData
from dal_toolbox import metrics
from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.models.utils.base import BaseModule
from dal_toolbox.models.utils.callbacks import MetricLogger, MetricHistory
from dal_toolbox.utils import seed_everything


@hydra.main(version_base=None, config_path="./configs", config_name="uncertainty")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    
    # Create output directory
    logging.info(f'Creating Output Directory at {args.output_dir}!')
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed everything for reporducability
    logging.info(f'Seeding for reproducability using random seed {args.random_seed}!')
    seed_everything(args.random_seed)

    # Build datasets
    logging.info(f'Building in-domain dataset {args.dataset} and ood-datasets {args.ood_datasets}!')
    dset = build_dataset(dataset=args.dataset, data_dir=args.data_dir)
    ood_dsets = build_ood_datasets(dsets=args.ood_datasets, data_dir=args.data_dir, transforms=dset.transforms)
    train_loader = DataLoader(dset.train_dataset, batch_size=args.model.train_batch_size, shuffle=True, drop_last=True)    
    test_loader_id = DataLoader(dset.test_dataset, batch_size=args.model.predict_batch_size)
    test_loaders_ood = {n: DataLoader(ood_ds, batch_size=args.model.predict_batch_size) for n, ood_ds in ood_dsets.items()}

    # Build model and trainer
    logging.info(f'Building the model {args.model.name}!')
    model = build_model(args, num_classes=dset.num_classes)
    metric_logger = MetricLogger()
    metric_history = MetricHistory()
    trainer = L.Trainer(
        max_epochs=args.model.n_epochs,
        callbacks=[metric_logger, metric_history],
        check_val_every_n_epoch=args.val_interval,
        enable_checkpointing=False,
        enable_progress_bar=False,
        devices=args.num_devices,
    )

    # Train the model
    logging.info(f"Start model-training.")
    trainer.fit(model, train_loader)

    # Evaluate the model in domain
    logging.info(f"Start in-domain evaluation.")
    test_predictions = trainer.predict(model, test_loader_id)
    logits = torch.cat([pred[0] for pred in test_predictions])
    targets = torch.cat([pred[1] for pred in test_predictions])
    id_test_stats = evaluate(logits, targets)
    logging.info('In-Domain Test Stats: %s', id_test_stats)

    # Evaluate the model out of domain
    logging.info(f"Start out-of-domain evaluation.")
    ood_test_stats = {}
    for name, ood_loader in test_loaders_ood.items():
        ood_predictions = trainer.predict(model, ood_loader)
        ood_logits = torch.cat([pred[0] for pred in ood_predictions])
        ood_results = evaluate_ood(id_logits=logits, ood_logits=ood_logits)
        ood_results = {f"{key}_{name}": val for key, val in ood_results.items()}
        ood_test_stats.update(ood_results)

    logging.info('Out-of-Domain Test Stats: %s', ood_test_stats)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logging.info(f"Saving results to {fname}.")
    results = {
        'train_stats':metric_history.metrics, 
        'id_test_stats': id_test_stats, 
        'ood_test_stats': ood_test_stats
        }
    with open(fname, 'w') as f:
        json.dump(results, f)




def evaluate(logits: Tensor, targets: Tensor) -> dict:
    """
    Calculates different metrics for in-domain evaluation.

    Args:
        logits (Tensor): Logits of in-domain samples.
        targets (Tensor): Targets of in-domain samples.

    Returns:
        test_stats (dic): Dictionary full of metrics.
    """

    logits = metrics.ensemble_log_softmax(logits) if logits.ndim == 3 else logits
    test_stats = {
        "accuracy": metrics.Accuracy()(logits, targets).item(),
        "nll": torch.nn.CrossEntropyLoss()(logits, targets).item(),
        "brier": metrics.BrierScore()(logits, targets).item(),
        "tce": metrics.ExpectedCalibrationError()(logits, targets).item(),
        "ace": metrics.AdaptiveCalibrationError()(logits, targets).item(),
    }
    return test_stats


def evaluate_ood(id_logits: Tensor, ood_logits: Tensor) -> dict:
    """
    Calculates differenn metrics for ood-evaluation.

    Args:
        id_logits (Tensor): Logits of in domain samples.
        ood_logits (Tensor): Logits of out-of-domain samples.

    Returns:
        test_stats (dic): Dictionary containing different evaluation stats.
    """
    test_stats = {}
    if ood_logits.ndim == 2:
        id_entropy = metrics.entropy_from_logits(id_logits)
        ood_entropy = metrics.entropy_from_logits(ood_logits)
    else:
        id_entropy = metrics.ensemble_entropy_from_logits(id_logits)
        ood_entropy = metrics.ensemble_entropy_from_logits(ood_logits)
    test_stats.update({
        f"aupr": metrics.OODAUPR()(id_entropy, ood_entropy).item(),
        f"auroc": metrics.OODAUROC()(id_entropy, ood_entropy).item()
    })
    return test_stats



def build_model(args: dict, **kwargs: dict) -> BaseModule:
    """
    Building the respective Module for the experiment.
    Args:
        args (dic): Arguments including settings for the module and training procedure
        kwargs (dic): Additional arguments.

    Returns:
        model (BaseModule): Deep Neural Network Model.
    """
    num_classes = kwargs['num_classes']

    def get_optim_and_lrs(params, args):
        optimizer = torch.optim.SGD(params=params,**args.model.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        return optimizer, lr_scheduler

    if args.model.name == 'deterministic':
        model = deterministic.resnet.ResNet18(num_classes=num_classes)
        optimizer, lr_scheduler = get_optim_and_lrs(params=model.parameters(), args=args)
        model = deterministic.DeterministicModel(model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'labelsmoothing':
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer, lr_scheduler = get_optim_and_lrs(params=model.parameters(), args=args)
        model = deterministic.DeterministicModel(model, loss_fn=nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing), optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'mixup':
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer, lr_scheduler = get_optim_and_lrs(params=model.parameters(), args=args)
        model = deterministic.DeterministicMixupModel(model, num_classes=num_classes, mixup_alpha=args.model.mixup_alpha, optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(num_classes, args.model.n_passes, args.model.dropout_rate)
        optimizer, lr_scheduler = get_optim_and_lrs(params=model.parameters(), args=args)
        model = mc_dropout.MCDropoutModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = deterministic.resnet.ResNet18(num_classes)
            opt, lrs = get_optim_and_lrs(params=mem.parameters(), args=args)
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = ensemble.EnsembleModel(model_list=members, optimizer_list=optimizers, lr_scheduler_list=lr_schedulers)
    elif args.model.name == 'sngp':
        # TODO: Will input_shape for resnet18_sngp change when using the imagenet-dataset?
        model = sngp.resnet.resnet18_sngp(num_classes=num_classes, input_shape=(3, 32, 32), **args.model.spectral_norm, **args.model.gp)
        optimizer, lr_scheduler = get_optim_and_lrs(params=model.parameters(), args=args)
        model = sngp.SNGPModel(model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=optimizer, lr_scheduler=lr_scheduler)
    else:
        raise NotImplementedError(f'{args.model.name} not implemented')

    return model


def build_dataset(dataset: str='CIFAR10', data_dir: str='./data/', val_split: float=0.) -> BaseData:
    """
    Building the in-domain dataset.

    Args:
        dataset (str): Name of the dataset.
        data_dir (str): Path of the dataset.

    Returns:
        dset (BaseData): The respective dataset.
    """
    if dataset == 'CIFAR10':
        dset = datasets.cifar.CIFAR10(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'CIFAR100':
        dset = datasets.cifar.CIFAR100(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'SVHN':
        dset = datasets.svhn.SVHN(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'Imagenet':
        dset = datasets.imagenet.ImageNet(dataset_path=data_dir, val_split=val_split)
    else:
        raise NotImplementedError()
    return dset


def build_ood_datasets(dsets: list=[], data_dir: str ='./data/', transforms: list=None) -> dict:
    """
    This method builds various datasets for out-of-distribution evaluation and extracts the respective test-data.

    Args:
        dsets (List): List containing dataset names for ood-evaluation.
        data_dir (str): Path to the respective datasets.
        transforms (List): List of transforms to apply to respective ood-datasets.

    Returns:
        ood_datasets (Dic): Dictionary containing the ood-datasets ready to use.
    """
    possible_ood_datasets = {
        "CIFAR10" : datasets.cifar.CIFAR10,
        "CIFAR100" : datasets.cifar.CIFAR100,
        "SVHN" : datasets.svhn.SVHN
        }
    ood_datasets = {}
    for dset in dsets:
        assert dset in possible_ood_datasets, f"The ood-datasets {dset} is not available!"
        data = possible_ood_datasets[dset](dataset_path=data_dir, transforms=transforms)
        ood_datasets[dset] = data.test_dataset
    return ood_datasets


if __name__ == '__main__':
    main()
