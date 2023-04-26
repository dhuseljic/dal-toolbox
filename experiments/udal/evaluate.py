"""This script evaluates an active learning cycle with indices that are given as arguments."""
import os
import time
import json
import logging

import hydra

from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.utils import seed_everything
from dal_toolbox.datasets import build_al_datasets
from active_learning import build_model


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}

    # Setup Dataset
    logging.info('Building datasets.')
    train_ds, query_ds, val_ds, ds_info = build_al_datasets(args)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    al_dataset = ALDataset(train_ds, query_ds, random_state=args.random_seed)

    logging.info('Loading initial labeled pool from %s', args.queried_indices_json)
    with open(args.queried_indices_json, 'r', encoding='utf-8') as f:
        queried_indices = json.load(f)
    al_dataset.update_annotations(queried_indices['cycle0'])
    n_init = len(queried_indices['cycle0'])
    acq_size = len(queried_indices['cycle1'])
    n_acq = len(queried_indices) - 1
    logging.info('Initial pool: %s  Acquisition Size: %s  Number of Acquisitions: %s', n_init, acq_size, n_acq)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_model(args, n_classes=ds_info['n_classes'])

    # Active Learning Cycles
    for i_acq in range(0, n_acq+1):
        logging.info('Starting AL iteration %s / %s', i_acq, n_acq)
        cycle_results = {}

        if i_acq != 0:
            indices = queried_indices[f'cycle{i_acq}']
            logging.info('Updating pool with %s samples', len(indices))
            al_dataset.update_annotations(indices)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        iter_per_epoch = len(al_dataset.labeled_dataset) // args.model.batch_size + 1
        train_sampler = RandomSampler(al_dataset.labeled_dataset, num_samples=args.model.batch_size*iter_per_epoch)
        train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, sampler=train_sampler)
        trainer.reset_states(reset_model=args.al_cycle.cold_start)
        history = trainer.train(args.model.n_epochs, train_loader)
        cycle_results['train_history'] = history['train_history']

        # Evaluate resulting model
        test_stats = trainer.evaluate(val_loader)
        cycle_results['test_stats'] = test_stats

        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })
        results[f'cycle{i_acq}'] = cycle_results

    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
