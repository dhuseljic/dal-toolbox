import os
import json
import logging
import torch
import hydra
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.distributed import destroy_process_group
from omegaconf import OmegaConf
import torchvision
from torchvision import transforms

from dal_toolbox.datasets.cifar import build_cifar10
from dal_toolbox.datasets.ssl_wrapper import PseudoLabelWrapper, PiModelWrapper, FixMatchWrapper, FlexMatchWrapper
from dal_toolbox.datasets.corruptions import RandAugment
from dal_toolbox.datasets.utils import sample_balanced_subset
from dal_toolbox.datasets.samplers import DistributedSampler
from dal_toolbox.models.deterministic.wide_resnet import wide_resnet_28_2
from dal_toolbox.utils import seed_everything, init_distributed_mode
from dal_toolbox.models.deterministic.trainer import DeterministicPseudoLabelTrainer, DeterministicPiModelTrainer, DeterministicFixMatchTrainer, DeterministicFlexMatchTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Check and initialize ddp if possible
    use_distributed = init_distributed_mode(args)
    if use_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        args.device = f'cuda:{rank}'

    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    misc = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataloaders for training and evaluation
    logging.info('Building datasets. Creating labeled pool with %s samples and \
        unlabeled pool with %s samples.', args.n_labeled_samples, args.n_unlabeled_samples)
    supervised_loader, unsupervised_loader, validation_loader, info = build_dataloaders(args, use_distributed)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_trainer(args, info['n_classes'], writer, use_distributed)

    # Training Process
    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        if use_distributed:
            supervised_loader.sampler.set_epoch(i_epoch)
            unsupervised_loader.sampler.set_epoch(i_epoch)

        # Train model for one epoch
        logging.info('Training epoch %s', i_epoch)
        train_stats = trainer.train_one_epoch(supervised_loader, unsupervised_loader, epoch=i_epoch)
        for key, value in train_stats.items():
            writer.add_scalar(tag=f"train/{key}", scalar_value=value, global_step=i_epoch)

        # Evaluate model on test set
        logging.info('Evaluation epoch %s', i_epoch)
        test_stats = trainer.evaluate(validation_loader)
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test/{key}", scalar_value=value, global_step=i_epoch)

        # Save results
        history_train.append(train_stats)
        history_test.append(test_stats)

    # Indices of torchvision dset are int64 which are not json compatible
    misc = {
        "labeled_indices": info['labeled_indices'],
    }

    results = {
        'train_history': history_train,
        'test_history': history_test,
        'misc': misc
    }

    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    with open(fname, 'w') as f:
        json.dump(results, f)

    if use_distributed:
        destroy_process_group()



def build_trainer(args, num_classes, summary_writer, use_distributed):
    model = wide_resnet_28_2(num_classes=num_classes, dropout_rate=args.model.dropout_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.model.optimizer.lr, momentum=0.9, weight_decay=args.model.optimizer.weight_decay, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_iter)
    criterion = torch.nn.CrossEntropyLoss()

    if args.ssl_algorithm.name == 'pseudo_labels':
        trainer = DeterministicPseudoLabelTrainer(
            model=model, n_classes=num_classes, n_iter=args.model.n_iter, p_cutoff=args.ssl_algorithm.p_cutoff, criterion=criterion,
            unsup_warmup=args.ssl_algorithm.unsup_warmup, lambda_u=args.ssl_algorithm.lambda_u, optimizer=optimizer, lr_scheduler=lr_scheduler, 
            device=args.device, output_dir=args.output_dir, summary_writer=summary_writer, use_distributed=use_distributed
        )
    elif args.ssl_algorithm.name == 'pi_model':
        trainer = DeterministicPiModelTrainer(
            model=model, n_classes=num_classes, n_iter=args.model.n_iter, unsup_warmup=args.ssl_algorithm.unsup_warmup, 
            optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler, device=args.device, output_dir=args.output_dir, 
            summary_writer=summary_writer, use_distributed=use_distributed, lambda_u=args.ssl_algorithm.lambda_u
        )
    elif args.ssl_algorithm.name == 'fixmatch':
        trainer = DeterministicFixMatchTrainer(
            model=model, n_classes=num_classes, n_iter=args.model.n_iter, p_cutoff=args.ssl_algorithm.p_cutoff,
            optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler, device=args.device, output_dir=args.output_dir, 
            summary_writer=summary_writer, use_distributed=use_distributed, lambda_u=args.ssl_algorithm.lambda_u
        )
    elif args.ssl_algorithm.name == 'flexmatch':
        trainer = DeterministicFlexMatchTrainer(
            model=model, n_classes=num_classes, n_iter=args.model.n_iter, p_cutoff=args.ssl_algorithm.p_cutoff,
            optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler, device=args.device, output_dir=args.output_dir, 
            summary_writer=summary_writer, use_distributed=use_distributed, lambda_u=args.ssl_algorithm.lambda_u, 
            ulb_ds_len=args.n_unlabeled_samples
        )
    else:
        assert True, 'algorithm not kown'
    return trainer



def build_ssl_dataset(args):
    if args.dataset == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
        ds = torchvision.datasets.CIFAR10(args.dataset_path, train=True, download=True)
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
        NotImplementedError()


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



def build_dataloaders(args, use_distributed):
    train_ds, info = build_cifar10('train', './data', return_info=True)
    train_ssl_ds = build_ssl_dataset(args)
    test_ds = build_cifar10('test', './data')

    labeled_indices = sample_balanced_subset(train_ds.targets, num_classes=info['n_classes'], num_samples=args.n_labeled_samples)
    train_ds_labeled = Subset(train_ds, labeled_indices)
    info['labeled_indices'] = labeled_indices

    Sampler = DistributedSampler if use_distributed else RandomSampler
    n_iter_per_epoch = args.model.n_iter // args.model.n_epochs

    supervised_sampler = Sampler(train_ds_labeled, num_samples=(n_iter_per_epoch * args.model.batch_size))
    unsupervised_sampler = Sampler(train_ssl_ds.ds, num_samples=int(n_iter_per_epoch * args.model.batch_size * args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    
    supervised_loader = DataLoader(train_ds_labeled, batch_size=args.model.batch_size, sampler=supervised_sampler)
    unsupervised_loader = DataLoader(train_ssl_ds, batch_size=int(args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=unsupervised_sampler)
    validation_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

    return supervised_loader, unsupervised_loader, validation_loader, info



if __name__ == "__main__":
    main()