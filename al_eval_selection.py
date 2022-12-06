# This script evaluates the sampling of an already run experiment
import os
import time
import copy
import json
import logging

import torch
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from active_learning.data import ALDataset
from models import build_model
from utils import seed_everything
from datasets import build_al_datasets


from active_learning.strategies import random, uncertainty, bayesian_uncertainty


@hydra.main(version_base=None, config_path="./configs", config_name="eval_selection")
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)
    results = {}


    logging.info('Loading results of finished experiment %s', args.result_json)
    with open(args.result_json, 'r') as f:
        data = json.load(f)

    logging.info('Building datasets.')
    train_ds, query_ds, val_ds, ds_info = build_al_datasets(args)
    al_dataset = ALDataset(train_ds, query_ds)

    logging.info('Building model: %s', args.model.name)
    model_dict = build_model(args, n_classes=ds_info['n_classes'], train_ds=al_dataset.labeled_dataset)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, lr_scheduler = model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    for i_acq, key in enumerate(data):
        cycle_results = {}

        #  If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_model_state)

        # Train model with labeled indices of other experiment
        loaded_results = data[key]
        al_dataset.load_state_dict({
            'labeled_indices': loaded_results['labeled_indices'],
            'unlabeled_indices': loaded_results['unlabeled_indices'],
        })
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        t1 = time.time()
        train_history = []
        for i_epoch in range(args.model.n_epochs):
            train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, shuffle=True, drop_last=True)
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
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
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
            "train_history": train_history,
            "test_stats": test_stats,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_cycle{i_acq}.pth'))

    # Save results
    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    # torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
    with open(fname, 'w') as f:
        json.dump(results, f)




    






if __name__ == "__main__":
    main()
