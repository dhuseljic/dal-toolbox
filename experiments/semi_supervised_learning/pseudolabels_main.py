import os
import json
import logging
import torch
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_ssl_dataset
from dal_toolbox.models import build_model
from dal_toolbox.utils import seed_everything

#TODO: Is unlabeled data with or without data augmentation?

@hydra.main(version_base=None, config_path="./configs", config_name="semi_supervised_learning")
def main(args):
    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    misc = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets. Creating labeled pool with %s samples and \
        unlabeled pool with %s samples.', args.n_labeled_samples, args.n_unlabeled_samples)
    lb_ds, ulb_ds, val_ds, ds_info = build_ssl_dataset(args)
    supervised_loader = DataLoader(lb_ds, batch_size=args.model.batch_size, shuffle=True)
    unsupervised_loader = DataLoader(ulb_ds, batch_size=int(args.model.batch_size*args.u_ratio), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    dataloaders = {
        "train_sup": supervised_loader,
        "train_unsup": unsupervised_loader
    }

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model_dict = build_model(args, n_classes=ds_info['n_classes'])
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    lr_scheduler = model_dict['lr_scheduler']

    # Training Process
    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        # Train model for one epoch
        logging.info('Training epoch %s', i_epoch)
        train_stats = train_one_epoch(
            model, dataloaders, **model_dict['train_kwargs'], epoch=i_epoch
        )
        if lr_scheduler:
            lr_scheduler.step()
        for key, value in train_stats.items():
            writer.add_scalar(tag=f"train/{key}", scalar_value=value, global_step=i_epoch)
        logging.info('Training stats: %s', train_stats)

        # Evaluate model on test set
        logging.info('Evaluation epoch %s', i_epoch)
        test_stats = evaluate(model, val_loader, dataloaders_ood={}, **model_dict['eval_kwargs'])
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test/{key}", scalar_value=value, global_step=i_epoch)
        logging.info('Evaluation stats: %s', test_stats)

        # Save results
        history_train.append(train_stats)
        history_test.append(test_stats)

    # Indices of torchvision dset are int64 which are not json compatible
    misc = {
        "labeled_indices": [int(i) for i in lb_ds.indices],
        "unlabeled_indices": [int(i) for i in ulb_ds.indices]
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


if __name__ == "__main__":
    main()