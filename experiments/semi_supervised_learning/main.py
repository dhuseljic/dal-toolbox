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

from dal_toolbox.datasets.cifar import build_cifar10
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
    trainloaders, testloaders, num_classes = build_dataloaders(args, use_distributed)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_trainer(args, num_classes, writer, use_distributed)

    # Training Process
    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        if use_distributed:
            for loader in trainloaders.values():
                loader.sampler.set_epoch(i_epoch)

        # Train model for one epoch
        logging.info('Training epoch %s', i_epoch)
        train_stats = trainer.train_one_epoch(**trainloaders, epoch=i_epoch)
        for key, value in train_stats.items():
            writer.add_scalar(tag=f"train/{key}", scalar_value=value, global_step=i_epoch)

        # Evaluate model on test set
        logging.info('Evaluation epoch %s', i_epoch)
        test_stats = trainer.evaluate(**testloaders)
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test/{key}", scalar_value=value, global_step=i_epoch)

        # Save results
        history_train.append(train_stats)
        history_test.append(test_stats)

    # Indices of torchvision dset are int64 which are not json compatible
    misc = {
        "labeled_indices": [int(i) for i in trainloaders['labeled_loader'].indices],
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



def build_dataloaders(args, use_distributed):
    train_ds_weak_aug, info = build_cifar10('ssl_weak', './data', return_info=True)
    num_classes = info['n_classes']
    train_ds_strong_aug = build_cifar10('ssl_strong', './data')
    test_ds = build_cifar10('test', './data')

    Sampler = DistributedSampler if use_distributed else RandomSampler
    n_iter_per_epoch = args.model.n_iter // args.model.n_epochs

    labeled_indices = sample_balanced_subset(train_ds_weak_aug.targets, num_classes, args.n_labeled_samples)
    train_ds_labeled = Subset(train_ds_weak_aug, labeled_indices)

    seed = args.random_seed + 12345
    g_1, g_2, g_3, g_4 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)

    supervised_sampler = Sampler(train_ds_labeled, num_samples=(n_iter_per_epoch * args.model.batch_size))
    random_sampler_weak_1 = Sampler(train_ds_weak_aug, num_samples=int(n_iter_per_epoch * args.model.batch_size * args.ssl_algorithm.u_ratio), generator=g_1)
    random_sampler_weak_2 = Sampler(train_ds_weak_aug, num_samples=int(n_iter_per_epoch * args.model.batch_size * args.ssl_algorithm.u_ratio), generator=g_2)
    random_sampler_strong = Sampler(train_ds_strong_aug, num_samples=int(n_iter_per_epoch * args.model.batch_size * args.ssl_algorithm.u_ratio), generator=g_3)
    random_sampler_idx = Sampler(range(len(train_ds_weak_aug)), num_samples=int(n_iter_per_epoch * args.model.batch_size * args.ssl_algorithm.u_ratio), generator=g_4)
    
    supervised_loader = DataLoader(train_ds_labeled, batch_size=args.model.batch_size, sampler=supervised_sampler)
    unsupervised_loader_weak_1 = DataLoader(train_ds_weak_aug, batch_size=int(args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_1)
    unsupervised_loader_weak_2 = DataLoader(train_ds_weak_aug, batch_size=int(args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_2)
    unsupervised_loader_strong = DataLoader(train_ds_strong_aug, batch_size=int(args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_strong)
    unsupervised_loader_idx = DataLoader(range(len(train_ds_weak_aug)), batch_size=int(args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_idx)
    val_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

    trainloaders, testloaders = {}, {}

    # Creating a Dataloader Dict
    if args.ssl_algorithm.name == 'fully_supervised':
        trainloaders['dataloader'] = supervised_loader
    else:
        trainloaders['labeled_loader'] = supervised_loader
        if args.ssl_algorithm.name == 'pseudo_labels':
            trainloaders['unlabeled_loader'] = unsupervised_loader_weak_1
        elif args.ssl_algorithm.name == 'pi_model':
            trainloaders['unlabeled_loader_weak_1'] = unsupervised_loader_weak_1
            trainloaders['unlabeled_loader_weak_2'] = unsupervised_loader_weak_2
        elif args.ssl_algorithm.name == 'fixmatch':
            trainloaders['unlabeled_loader_weak'] = unsupervised_loader_weak_1
            trainloaders['unlabeled_loader_strong'] = unsupervised_loader_strong
        elif args.ssl_algorithm.name == 'flexmatch':
            trainloaders['unlabeled_loader_weak'] = unsupervised_loader_weak_1
            trainloaders['unlabeled_loader_strong'] = unsupervised_loader_strong
            trainloaders['unlabeled_loader_indices'] = unsupervised_loader_idx
        else:
            assert True, 'No valid ssl_algorithm chosen!'
    testloaders['dataloader'] = val_loader

    return trainloaders, testloaders, num_classes


def sample_balanced_subset(targets, num_classes, num_samples):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # Get samples per class
    assert num_samples % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
    lb_samples_per_class = [int(num_samples / num_classes)] * num_classes

    val_pool = []
    for c in range(num_classes):
        idx = np.array([i for i in range(len(targets)) if targets[i] == c])
        np.random.shuffle(idx)
        val_pool.extend(idx[:lb_samples_per_class[c]])
    return [int(i) for i in val_pool]


if __name__ == "__main__":
    main()