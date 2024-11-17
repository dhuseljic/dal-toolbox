import os
import json
import logging
import torch
import hydra
import lightning as L

from torch.utils.data import DataLoader, RandomSampler, Subset
from omegaconf import OmegaConf
from dal_toolbox import metrics
from dal_toolbox.datasets import cifar
from dal_toolbox.datasets.utils import sample_balanced_subset
from dal_toolbox.models.deterministic import resnet, DeterministicModel
from dal_toolbox.models.deterministic.base_semi import DeterministicPseudoLabelModel, DeterministicPiModel, DeterministicFixMatchModel
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger, MetricHistory



@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Initial Setup
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Dataloaders for training and evaluation
    logging.info('Building datasets.')
    data, ssl_data = build_datasets(args)
    logging.info('Creating labeled pool with %s instances per class.', args.num_labeled // data.num_classes)
    logging.info('Creating unlabeled pool with %s instances.', args.num_unlabeled)
    supervised_loader, unsupervised_loader = build_dataloaders(args, data.train_dataset, ssl_data.train_dataset)
    val_loader = DataLoader(data.val_dataset, batch_size=args.model.predict_batch_size, shuffle=False)
    test_loader = DataLoader(data.test_dataset, batch_size=args.model.predict_batch_size, shuffle=False)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes)

    metric_logger = MetricLogger()
    metric_history = MetricHistory()

    trainer = L.Trainer(
        max_steps=args.model.num_iter,
        callbacks=[metric_logger, metric_history],
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=args.output_dir,
        check_val_every_n_epoch=args.val_interval,
        logger=False,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(
        model,
        train_dataloaders=supervised_loader if args.ssl_algorithm.name == 'fully_supervised' else {'labeled': supervised_loader, 'unlabeled': unsupervised_loader},
        val_dataloaders=val_loader,
    )

    test_stats = trainer.validate(model, test_loader)[0]
    logging.info('Test stats: %s', test_stats)

    results = {
        'train_stats' : metric_history.metrics, 
        'test_stats' : test_stats, 
        }

    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    with open(fname, 'w') as f:
        json.dump(results, f)


def build_model(args, num_classes):
    model = resnet.ResNet18(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_iter)
    if args.ssl_algorithm.name == 'fully_supervised':
        model = DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={
                                              'val_acc': metrics.Accuracy()})
    elif args.ssl_algorithm.name == 'pseudo_labels':
        model = DeterministicPseudoLabelModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={
                                              'val_acc': metrics.Accuracy()})
    elif args.ssl_algorithm.name == 'pi_model':
        model = DeterministicPiModel(model, num_classes=num_classes, optimizer=optimizer,
                                     lr_scheduler=lr_scheduler, val_metrics={'val_acc': metrics.Accuracy()})
    elif args.ssl_algorithm.name == 'fixmatch':
        model = DeterministicFixMatchModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={
                                           'val_acc': metrics.Accuracy()})
    return model


def build_datasets(args):
    if args.dataset == 'CIFAR10':
        if args.ssl_algorithm.name == 'fully_supervised':
            ssl_transforms = cifar.CIFAR10StandardTransforms()
        elif args.ssl_algorithm.name == 'pseudo_labels':
            ssl_transforms = cifar.CIFAR10PseudoLabelTransforms()
        elif args.ssl_algorithm.name == 'pi_model':
            ssl_transforms = cifar.CIFAR10PIModelTransforms()
        elif args.ssl_algorithm.name == 'fixmatch':
            ssl_transforms = cifar.CIFAR10FixMatchTransforms()
        else:
            raise NotImplementedError

        data = cifar.CIFAR10(args.data_dir, seed=args.random_seed)
        ssl_data = cifar.CIFAR10(args.data_dir, transforms=ssl_transforms, seed=args.random_seed)
    else:
        raise NotImplementedError()
    return data, ssl_data


def build_dataloaders(args, train_ds, train_ds_ssl):
    targets = torch.Tensor([batch[-1] for batch in train_ds]).long()
    labeled_indices = sample_balanced_subset(targets, num_samples=args.num_labeled)
    labeled_ds = Subset(train_ds, labeled_indices)

    if args.ssl_algorithm.name == 'fully_supervised':
        return DataLoader(labeled_ds, batch_size=args.model.train_batch_size, shuffle=True, drop_last=(args.model.train_batch_size<args.num_labeled)), None
    else:
        num_iter_per_epoch = args.model.num_iter // args.model.num_epochs

        labeled_sampler = RandomSampler(labeled_ds, num_samples=(num_iter_per_epoch * args.model.train_batch_size))
        labeled_loader = DataLoader(labeled_ds, batch_size=args.model.train_batch_size, sampler=labeled_sampler)

        unlabeled_num_samples = int(num_iter_per_epoch * args.model.train_batch_size * args.ssl_algorithm.u_ratio)
        unlabeled_sampler = RandomSampler(train_ds_ssl, num_samples=unlabeled_num_samples)
        unlabeled_batch_size = int(args.model.train_batch_size*args.ssl_algorithm.u_ratio)
        unlabeled_loader = DataLoader(train_ds_ssl, batch_size=unlabeled_batch_size, sampler=unlabeled_sampler)

        return labeled_loader, unlabeled_loader


if __name__ == "__main__":
    main()