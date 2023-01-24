import os
import json
import logging
import hydra
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_ssl_dataset
from dal_toolbox.models import wide_resnet
from dal_toolbox.utils import seed_everything


@hydra.main(version_base=None, config_path="./configs", config_name="pi_model")
def main(args):
    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets. Creating labeled pool with %s samples and unlabeled pool with %s samples.',
                 args.n_labeled_samples, args.n_unlabeled_samples)
    lb_ds, ulb_ds_weak, _, val_ds, ds_info = build_ssl_dataset(args)

    # Setup dataloaders
    n_iter_per_epoch = args.model.n_iter // args.model.n_epochs
    supervised_sampler = RandomSampler(lb_ds, num_samples=(n_iter_per_epoch * args.model.batch_size))
    supervised_loader = DataLoader(lb_ds, batch_size=args.model.batch_size, sampler=supervised_sampler)

    random_sampler_weak_1 = RandomSampler(ulb_ds_weak, num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                          args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    random_sampler_weak_2 = RandomSampler(ulb_ds_weak, num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                          args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    unsupervised_loader_weak_1 = DataLoader(ulb_ds_weak, batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_1)
    unsupervised_loader_weak_2 = DataLoader(ulb_ds_weak, batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_2)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    dataloaders = {
        "train_sup": supervised_loader,
        "train_unsup_weak_1": unsupervised_loader_weak_1,
        "train_unsup_weak_2": unsupervised_loader_weak_2
    }

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = wide_resnet.WideResNet(28, 2, dropout_rate=0, num_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model.optimizer.lr,
        weight_decay=args.model.optimizer.weight_decay,
        momentum=args.model.optimizer.momentum,
        nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)

    # Training Process
    logging.info('Starting training.')
    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        # Train model for one epoch
        train_stats = wide_resnet.train_one_epoch_pimodel(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            n_iter=args.model.n_iter,
            lambda_u=args.ssl_algorithm.lambda_u,
            unsup_warmup=args.ssl_algorithm.unsup_warmup,
            device=args.device,
            epoch=i_epoch,
        )
        lr_scheduler.step()
        for key, value in train_stats.items():
            writer.add_scalar(tag=f"train/{key}", scalar_value=value, global_step=i_epoch)
        # logging.info('Training stats: %s', train_stats)
        history_train.append(train_stats)

        if (i_epoch+1) % args.eval_interval == 0 or (i_epoch+1) == args.model.n_epochs:
            # Evaluate model on test set
            logging.info('Evaluation epoch %s', i_epoch)
            test_stats = wide_resnet.evaluate(model, val_loader, dataloaders_ood={},
                                              criterion=criterion, device=args.device)
            for key, value in test_stats.items():
                writer.add_scalar(tag=f"test/{key}", scalar_value=value, global_step=i_epoch)
            logging.info('Evaluation stats: %s', test_stats)
            history_test.append(test_stats)

    results = {
        'train_history': history_train,
        'test_history': history_test,
        "labeled_indices": [int(i) for i in lb_ds.indices],
        "unlabeled_indices": [int(i) for i in ulb_ds_weak.indices],
    }

    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    with open(fname, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
