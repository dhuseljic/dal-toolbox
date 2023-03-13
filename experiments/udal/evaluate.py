"""This script evaluates an active learning cycle with indices that are given as arguments."""
import os
import time
import copy
import json
import logging

import torch
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from omegaconf import OmegaConf

from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.models import build_model
from dal_toolbox.utils import seed_everything
from dal_toolbox.datasets import build_al_datasets
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge, predefined


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    writer = SummaryWriter(log_dir=args.output_dir)

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
    model_dict = build_model(args, n_classes=ds_info['n_classes'])
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, lr_scheduler = model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    # Active Learning Cycles
    for i_acq in range(0, n_acq+1):
        logging.info('Starting AL iteration %s / %s', i_acq, n_acq)
        cycle_results = {}

        if i_acq != 0:
            indices = queried_indices[f'cycle{i_acq}']
            logging.info('Updating pool with %s samples', len(indices))
            al_dataset.update_annotations(indices)


        #  If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        if args.cold_start:
            model.load_state_dict(initial_model_state)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        t1 = time.time()
        iter_per_epoch = len(al_dataset.labeled_dataset) // args.model.batch_size + 1
        train_sampler = RandomSampler(al_dataset.labeled_dataset, num_samples=args.model.batch_size*iter_per_epoch)
        train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, sampler=train_sampler)
        train_history = []

        for i_epoch in range(args.model.n_epochs):
            train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
            if lr_scheduler:
                lr_scheduler.step()

            for key, value in train_stats.items():
                writer.add_scalar(tag=f"cycle_{i_acq}_train/{key}", scalar_value=value, global_step=i_epoch)
            train_history.append(train_stats)
        training_time = (time.time() - t1)
        logging.info('Training took %.2f minutes', training_time/60)
        logging.info('Training stats: %s', train_stats)
        cycle_results['train_history'] = train_history
        cycle_results['training_time'] = training_time

        # Evaluate resulting model
        logging.info('Evaluation with %s samples', len(val_ds))
        t1 = time.time()
        test_stats = evaluate(model, val_loader, dataloaders_ood={}, **model_dict['eval_kwargs'])
        evaluation_time = time.time() - t1
        logging.info('Evaluation took %.2f minutes', evaluation_time/60)
        logging.info('Evaluation stats: %s', test_stats)
        cycle_results['evaluation_time'] = evaluation_time
        cycle_results['test_stats'] = test_stats

        # Log
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test_stats/{key}", scalar_value=value, global_step=i_acq)

        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })
        results[f'cycle{i_acq}'] = cycle_results

        # Save checkpoint
        logging.info('Saving checkpoint for cycle %s', i_acq)
        checkpoint = {
            "args": args,
            "model": model.state_dict(),
            "al_dataset": al_dataset.state_dict(),
            "optimizer": model_dict['train_kwargs']['optimizer'].state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "cycle_results": cycle_results,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Saving
    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Save Model
    file_name = os.path.join(args.output_dir, "model_final.pth")
    logging.info("Saving final model to %s.", file_name)
    torch.save(checkpoint, file_name)


if __name__ == "__main__":
    main()
