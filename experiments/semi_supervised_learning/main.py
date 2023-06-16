import os
import json
import logging
import torch
import hydra
import lightning as L

from torch.utils.data import DataLoader, RandomSampler, Subset
from omegaconf import OmegaConf
from torchvision import transforms
from dal_toolbox import metrics
from dal_toolbox.datasets.cifar import build_cifar10
from dal_toolbox.datasets.ssl_wrapper import PseudoLabelWrapper, PiModelWrapper, FixMatchWrapper, FlexMatchWrapper
from dal_toolbox.datasets.corruptions import RandAugment
from dal_toolbox.datasets.utils import sample_balanced_subset
from dal_toolbox.models.deterministic import resnet
from dal_toolbox.models.deterministic.base_semi import DeterministicPseudoLabelModel, DeterministicPiModel, DeterministicFixMatchModel
from dal_toolbox.utils import seed_everything
# from dal_toolbox.models.deterministic.trainer import DeterministicPseudoLabelTrainer, DeterministicPiModelTrainer, DeterministicFixMatchTrainer, DeterministicFlexMatchTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Dataloaders for training and evaluation
    logging.info('Building datasets. Creating labeled pool with %s samples and \
        unlabeled pool with %s samples.', args.n_labeled_samples, args.n_unlabeled_samples)
    supervised_loader, unsupervised_loader, val_loader, info = build_dataloaders(args)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = resnet.ResNet18(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_iter)
    # model = DeterministicPseudoLabelModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={'val_acc': metrics.Accuracy()})
    # model = DeterministicPiModel(model, num_classes=10, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={'val_acc': metrics.Accuracy()})
    model = DeterministicFixMatchModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler, val_metrics={'val_acc': metrics.Accuracy()})

    trainer = L.Trainer(
        max_steps=args.model.num_iter,
        enable_checkpointing=False,
        default_root_dir=args.output_dir,
    )
    trainer.fit(
        model,
        train_dataloaders={'labeled': supervised_loader, 'unlabeled': unsupervised_loader},
        val_dataloaders=val_loader,
    )

    results = {
    }

    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    with open(fname, 'w') as f:
        json.dump(results, f)


def build_model(args, num_classes):
    pass


def build_ssl_dataset(args):
    if args.dataset == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
        ds = build_cifar10('raw', args.dataset_path)
        transform_weak = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_strong = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise NotImplementedError()

    if args.ssl_algorithm.name == 'pseudo_labels':
        ds = PseudoLabelWrapper(
            ds=ds,
            ds_path=args.dataset_path,
            transforms_weak=transform_weak,
            transforms_strong=transform_strong
        )
    elif args.ssl_algorithm.name == 'pi_model':
        ds = PiModelWrapper(
            ds=ds,
            ds_path=args.dataset_path,
            transforms_weak=transform_weak,
            transforms_strong=transform_strong
        )
    elif args.ssl_algorithm.name == 'fixmatch':
        ds = FixMatchWrapper(
            ds=ds,
            ds_path=args.dataset_path,
            transforms_weak=transform_weak,
            transforms_strong=transform_strong
        )
    elif args.ssl_algorithm.name == 'flexmatch':
        ds = FlexMatchWrapper(
            ds=ds,
            ds_path=args.dataset_path,
            transforms_weak=transform_weak,
            transforms_strong=transform_strong
        )
    else:
        assert True, 'algorithm not kown'
    return ds


def build_dataloaders(args):
    train_ds, info = build_cifar10('train', './data', return_info=True)
    train_ssl_ds = build_ssl_dataset(args)
    test_ds = build_cifar10('test', './data')

    labeled_indices = sample_balanced_subset(
        train_ds.targets, num_classes=info['n_classes'], num_samples=args.n_labeled_samples)
    train_ds_labeled = Subset(train_ds, labeled_indices)
    info['labeled_indices'] = labeled_indices

    # Sampler = DistributedSampler if use_distributed else RandomSampler
    Sampler = RandomSampler
    n_iter_per_epoch = args.model.num_iter // args.model.num_epochs

    supervised_sampler = Sampler(train_ds_labeled, num_samples=(n_iter_per_epoch * args.model.train_batch_size))
    unsupervised_sampler = Sampler(train_ssl_ds.ds, num_samples=int(
        n_iter_per_epoch * args.model.train_batch_size * args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))

    supervised_loader = DataLoader(train_ds_labeled, batch_size=args.model.train_batch_size, sampler=supervised_sampler)
    unsupervised_loader = DataLoader(train_ssl_ds, batch_size=int(
        args.model.train_batch_size*args.ssl_algorithm.u_ratio), sampler=unsupervised_sampler)
    validation_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

    return supervised_loader, unsupervised_loader, validation_loader, info


if __name__ == "__main__":
    main()
