import os
import json
import logging

import hydra
import torch
import torch.nn as nn
import lightning as L

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything, is_running_on_slurm


@hydra.main(version_base=None, config_path="./configs", config_name="uncertainty")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    misc = {}

    # Load data
    data = build_dataset(args)
    train_loader = DataLoader(data.train_dataset, batch_size=args.model.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(data.val_dataset, batch_size=args.model.predict_batch_size)

    test_dataset = data.test_dataset
    ood_datasets = build_ood_datasets(args, data.mean, data.mean)
    test_loader_id = DataLoader(test_dataset, batch_size=args.model.predict_batch_size)
    test_loaders_ood = {n: DataLoader(ood_ds, batch_size=args.model.predict_batch_size)
                        for n, ood_ds in ood_datasets.items()}

    # Training
    logger.info('Starting Training..')
    model = build_model(args, num_classes=data.num_classes)
    callbacks = []
    if is_running_on_slurm():
        callbacks.append(MetricLogger())
    trainer = L.Trainer(
        max_epochs=args.model.n_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        enable_checkpointing=False,
        enable_progress_bar=is_running_on_slurm() is False,
        devices=args.num_devices,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    # Evaluation
    test_predictions = trainer.predict(model, test_loader_id)
    logits = torch.cat([pred[0] for pred in test_predictions])
    targets = torch.cat([pred[1] for pred in test_predictions])

    test_stats = evaluate(logits, targets)

    for name, ood_loader in test_loaders_ood.items():
        ood_predictions = trainer.predict(model, ood_loader)
        ood_logits = torch.cat([pred[0] for pred in ood_predictions])
        ood_results = evaluate_ood(id_logits=logits, ood_logits=ood_logits)
        ood_results = {f"{key}_{name}": val for key, val in ood_results.items()}
        test_stats.update(ood_results)

    logging.info('Test Stats: %s', test_stats)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logger.info("Saving results to %s", fname)
    results = {'test_stats': test_stats, 'misc': misc}
    with open(fname, 'w') as f:
        json.dump(results, f)


def evaluate(logits, targets):
    test_stats = {}

    # Model specific test loss and accuracy for in domain testset
    logits = metrics.ensemble_log_softmax(logits) if logits.ndim == 3 else logits

    # Test stats for in-distribution
    test_stats.update({
        "accuracy": metrics.Accuracy()(logits, targets).item(),
        "nll": torch.nn.CrossEntropyLoss()(logits, targets).item(),
        "brier": metrics.BrierScore()(logits, targets).item(),
        "tce": metrics.ExpectedCalibrationError()(logits, targets).item(),
        "ace": metrics.AdaptiveCalibrationError()(logits, targets).item(),
    })
    return test_stats


def evaluate_ood(id_logits, ood_logits):
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


def build_model(args, **kwargs):
    num_classes = kwargs['num_classes']

    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(num_classes=num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model = deterministic.DeterministicModel(
            model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )

    elif args.model.name == 'resnet18_labelsmoothing':
        # Lightning:
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model = deterministic.DeterministicModel(
            model,
            loss_fn=nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )

    elif args.model.name == 'resnet18_mixup':
        # Lightning:
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model = deterministic.DeterministicMixupModel(
            model,
            num_classes=num_classes,
            mixup_alpha=args.model.mixup_alpha,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )

    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(num_classes, args.model.n_passes, args.model.dropout_rate)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model = mc_dropout.MCDropoutModel(
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics={'val_acc': metrics.Accuracy()},
        )

    elif args.model.name == 'resnet18_ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = deterministic.resnet.ResNet18(num_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
            )
            lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.model.n_epochs)
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = ensemble.EnsembleModel(
            model_list=members,
            optimizer_list=optimizers,
            lr_scheduler_list=lr_schedulers,
        )

    elif args.model.name == 'resnet18_sngp':
        model = sngp.resnet.resnet18_sngp(
            num_classes=num_classes,
            input_shape=(3, 32, 32),
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.norm_bound,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=False,
            random_feature_type=args.model.gp.random_feature_type,
            scale_random_features=args.model.gp.scale_random_features,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model = sngp.SNGPModel(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    else:
        raise NotImplementedError(f'{args.model.name} not implemented')

    return model


def build_dataset(args):
    if args.dataset == 'CIFAR10':
        data = datasets.cifar.CIFAR10(args.dataset_path)
        return data

    elif args.dataset == 'CIFAR100':
        data = datasets.cifar.CIFAR100(args.dataset_path)

    elif args.dataset == 'SVHN':
        data = datasets.svhn.SVHN(args.dataset_path)

    elif args.dataset == 'Imagenet':
        data = datasets.imagenet.ImageNet(args.dataset_path)

    else:
        raise NotImplementedError()

    return data


def build_ood_datasets(args, mean, std):
    ood_datasets = {}
    if 'CIFAR10' in args.ood_datasets:
        data = datasets.cifar.CIFAR10(args.dataset_path, mean=mean, std=std)
        ood_datasets["CIFAR10"] = data.test_dataset

    if 'CIFAR100' in args.ood_datasets:
        data = datasets.cifar.CIFAR100(args.dataset_path, mean=mean, std=std)
        ood_datasets["CIFAR100"] = data.test_dataset

    if 'SVHN' in args.ood_datasets:
        data = datasets.svhn.SVHN(args.dataset_path, mean=mean, std=std)
        ood_datasets["SVHN"] = data.test_dataset

    return ood_datasets


if __name__ == '__main__':
    main()
