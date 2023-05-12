import os
import json
import logging

import hydra
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, RandomSampler

from dal_toolbox import datasets
from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp, variational_inference
from dal_toolbox.utils import seed_everything
# from dal_toolbox.models.utils.callbacks import DummyCallback


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

    # Load model
    logger.info('Starting Training..')

    # Own trainer
    model, trainer = build_model(args, n_classes=ds_info['n_classes'])
    trainer.fit(train_loader, val_loaders=test_loader_id)
    logger.info('Starting Testing..')
    test_stats = trainer.evaluate(test_loader_id, dataloaders_ood=test_loaders_ood)
    logger.info("Final test results: %s", test_stats)

    # Lightning:
    # model = get_lightning_model()
    # trainer = L.Trainer(
    #     max_epochs=args.model.n_epochs,
    #     callbacks=[Logger()],
    #     check_val_every_n_epoch=args.eval_interval,
    #     enable_checkpointing=False,
    #     enable_progress_bar=False,
    #     devices=args.num_devices,
    # )
    # trainer.fit(model, train_loader, val_dataloaders=test_loader_id)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logger.info("Saving results to %s", fname)
    results = {'test_stats': test_stats, 'misc': misc}
    with open(fname, 'w') as f:
        json.dump(results, f)


def build_model(args, **kwargs):
    num_classes = kwargs['n_classes']

    if args.model.name == 'resnet18_deterministic':
        # Lightning: model = deterministic.resnet.ResNet18(n_classes)
        # Fabric:
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        trainer = deterministic.trainer.DeterministicTrainer(
            model,
            nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs),
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
            output_dir=args.output_dir,
            # callbacks=[DummyCallback()]
        )
        return model, trainer

    elif args.model.name == 'resnet18_labelsmoothing':
        # Lightning:
        # model = deterministic.resnet.ResNet18Labelsmoothing(n_classes, label_smoothing=args.model.label_smoothing)
        # Fabric:
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        trainer = deterministic.trainer.DeterministicTrainer(
            model,
            nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing),
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs),
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
            output_dir=args.output_dir,
        )
        return model, trainer

    elif args.model.name == 'resnet18_mixup':
        # Lightning:
        # model = deterministic.resnet.ResNet18Mixup(n_classes, mixup_alpha=.1)
        # Fabric:
        model = deterministic.resnet.ResNet18(num_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        trainer = deterministic.trainer.DeterministicMixupTrainer(
            model,
            nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs),
            num_classes=num_classes,
            mixup_alpha=args.model.mixup_alpha,
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
            output_dir=args.output_dir,
        )
        return model, trainer

    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(num_classes, args.model.n_passes, args.model.dropout_rate)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            momentum=args.model.optimizer.momentum,
            weight_decay=args.model.optimizer.weight_decay,
        )
        trainer = mc_dropout.trainer.MCDropoutTrainer(
            model,
            nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs),
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
            output_dir=args.output_dir,
        )
        return model, trainer

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
        model = ensemble.voting_ensemble.Ensemble(members)
        criterion = nn.CrossEntropyLoss()
        optimizer = ensemble.voting_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = ensemble.voting_ensemble.EnsembleLRScheduler(lr_schedulers)
        trainer = ensemble.trainer.EnsembleTrainer(
            model,
            nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
            output_dir=args.output_dir,
        )
        return model, trainer

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
        criterion = nn.CrossEntropyLoss()
        trainer = sngp.trainer.SNGPTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            output_dir=args.output_dir,
            num_epochs=args.model.n_epochs,
            num_devices=args.num_devices,
        )
        return model, trainer

    elif args.model.name == 'resnet18_vi':
        model = variational_inference.resnet.BayesianResNet18(num_classes=10, prior_sigma=args.model.vi.prior_sigma)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        criterion = nn.CrossEntropyLoss()
        trainer = variational_inference.trainer.VITrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            grad_norm=args.model.optimizer.grad_norm,
            lr_scheduler=lr_scheduler,
            mc_samples=args.model.vi.mc_samples,
            kl_temperature=args.model.vi.kl_temperature,
            device=args.device,
            output_dir=args.output_dir
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
