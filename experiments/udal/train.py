import os
import json
import logging

import torch
import hydra

from torch.utils.data import Subset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from active_learning import build_model, build_datasets
from dal_toolbox.utils import seed_everything


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets.')
    dataset, _, _, ds_info = build_datasets(args)
    n_train_samples = int(len(dataset) * (1 - args.val_split))
    indices = torch.randperm(len(dataset))
    train_indices, val_indices = indices[:n_train_samples], indices[n_train_samples:]
    train_indices = train_indices[:args.budget]
    train_ds = Subset(dataset, indices=train_indices)
    val_ds = Subset(dataset, indices=val_indices)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_model(args, n_classes=ds_info['n_classes'])

    # Train
    logging.info('Training on dataset with %s samples.', len(train_indices))
    iter_per_epoch = len(train_ds) // args.model.batch_size + 1
    train_sampler = RandomSampler(train_ds, num_samples=args.model.batch_size*iter_per_epoch)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.model.batch_size)
    history = trainer.train(args.model.n_epochs, train_loader)
    results['train_history'] = history['train_history'] 

    logging.info('Validation on dataset with %s samples.', len(val_indices))
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    val_stats = trainer.evaluate(val_loader)
    logging.info('Validation stats: %s', val_stats)
    results['val_stats'] = val_stats

    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
