import os
import json
import logging

import hydra
import torch
import torch.nn as nn
import lightning as L

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, RandomSampler

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
    train_ds, test_ds_id, ds_info = build_dataset(args)
    ood_datasets = build_ood_datasets(args, ds_info['mean'], ds_info['std'])
    if args.num_samples:
        logger.info('Creating random training subset with %s samples. Saving indices.', args.num_samples)
        indices_id = torch.randperm(len(train_ds))[:args.num_samples]
        train_ds = Subset(train_ds, indices=indices_id)
        misc['train_indices'] = indices_id.tolist()

    logger.info('Training on %s with %s samples.', args.dataset, len(train_ds))
    logger.info('Test in-distribution dataset %s has %s samples.', args.dataset, len(test_ds_id))
    for name, test_ds_ood in ood_datasets.items():
        logger.info('Test out-of-distribution dataset %s has %s samples.', name, len(test_ds_ood))

    # Prepare dataloaders
    iter_per_epoch = len(train_ds) // args.model.batch_size + 1
    train_sampler = RandomSampler(train_ds, num_samples=(iter_per_epoch * args.model.batch_size))
    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, sampler=train_sampler)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.test_batch_size)
    test_loaders_ood = {name: DataLoader(test_ds_ood, batch_size=args.test_batch_size)
                        for name, test_ds_ood in ood_datasets.items()}

    # Training
    logger.info('Starting Training..')
    model = build_model(args, num_classes=ds_info['n_classes'])
    callbacks = []
    if is_running_on_slurm():
        callbacks.append(MetricLogger())
    trainer = L.Trainer(
        max_epochs=args.model.n_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=args.eval_interval,
        enable_checkpointing=False,
        enable_progress_bar=is_running_on_slurm() is False,
        devices=args.num_devices,
    )
    trainer.fit(model, train_loader)  # , val_dataloaders=test_loader_id)

    # Evaluation
    predictions_id = trainer.predict(model, test_loader_id)
    logits_id = torch.cat([pred[0] for pred in predictions_id])
    targets_id = torch.cat([pred[1] for pred in predictions_id])

    predictions_ood = trainer.predict(model, test_loaders_ood)
    logits_ood_dict = {}
    for ds_name, predictions in zip(test_loaders_ood, predictions_ood):
        logits_ood_dict[ds_name] = torch.cat([pred[0] for pred in predictions])

    test_stats = evaluate(logits_id, targets_id, logits_ood_dict)
    logging.info('Test Stats: %s', test_stats)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logger.info("Saving results to %s", fname)
    results = {'test_stats': test_stats, 'misc': misc}
    with open(fname, 'w') as f:
        json.dump(results, f)


def evaluate(logits_id, targets_id, logits_ood_dict=None):
    # Model specific test loss and accuracy for in domain testset
    test_stats = {}
    logits_id = metrics.ensemble_log_probas_from_logits(logits_id) if logits_id.ndim == 3 else logits_id

    # Test stats for in-distribution
    test_stats.update({
        "accuracy": metrics.Accuracy()(logits_id, targets_id).item(),
        "nll": torch.nn.CrossEntropyLoss()(logits_id, targets_id).item(),
        "brier": metrics.BrierScore()(logits_id, targets_id).item(),
        "tce": metrics.ExpectedCalibrationError()(logits_id, targets_id).item(),
        "ace": metrics.AdaptiveCalibrationError()(logits_id, targets_id).item(),
    })

    # Test stats for out-of-distribution
    entropy_id = metrics.entropy_from_logits(logits_id)
    for ds_name, logits_ood in logits_ood_dict.items():
        if logits_ood.ndim == 2:
            entropy_ood = metrics.entropy_from_logits(logits_ood)
        else:
            entropy_ood = metrics.ensemble_entropy_from_logits(logits_ood)
        test_stats.update({
            f"aupr_{ds_name}": metrics.OODAUPR()(entropy_id, entropy_ood).item(),
            f"auroc_{ds_name}": metrics.OODAUROC()(entropy_id, entropy_ood).item()
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
            train_metrics={'train_acc': metrics.Accuracy()}
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
            train_metrics={'train_acc': metrics.Accuracy()}
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
        train_ds, ds_info = datasets.cifar.build_cifar10('train', args.dataset_path, return_info=True)
        test_ds = datasets.cifar.build_cifar10('test', args.dataset_path)

    elif args.dataset == 'CIFAR100':
        train_ds, ds_info = datasets.cifar.build_cifar100('train', args.dataset_path, return_info=True)
        test_ds = datasets.cifar.build_cifar100('test', args.dataset_path)

    elif args.dataset == 'SVHN':
        train_ds, ds_info = datasets.svhn.build_svhn('train', args.dataset_path, return_info=True)
        test_ds = datasets.svhn.build_svhn('test', args.dataset_path)

    elif args.dataset == 'Imagenet':
        train_ds, ds_info = datasets.imagenet.build_imagenet('train', args.dataset_path, return_info=True)
        test_ds = datasets.imagenet.build_imagenet('val', args.dataset_path)
    else:
        raise NotImplementedError

    return train_ds, test_ds, ds_info


def build_ood_datasets(args, mean, std):
    ood_datasets = {}
    if 'CIFAR10' in args.ood_datasets:
        test_ds_ood = datasets.cifar.build_cifar10('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR10"] = test_ds_ood

    if 'CIFAR100' in args.ood_datasets:
        test_ds_ood = datasets.cifar.build_cifar100('test', args.dataset_path, mean, std)
        ood_datasets["CIFAR100"] = test_ds_ood

    if 'SVHN' in args.ood_datasets:
        test_ds_ood = datasets.svhn.build_svhn('test', args.dataset_path, mean, std)
        ood_datasets["SVHN"] = test_ds_ood

    return ood_datasets


if __name__ == '__main__':
    main()
