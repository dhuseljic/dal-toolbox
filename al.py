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


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    # logging.basicConfig(filename=os.path.join(args.output_dir, 'active_learning.log'), filemode='w')
    # logging.basicConfig(filename=os.path.join(args.output_dir, 'active_learning.log'), filemode='w')
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets. Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
    train_ds, query_ds, val_ds, ds_info = build_al_datasets(args)
    al_dataset = ALDataset(train_ds, query_ds)
    al_dataset.random_init(n_samples=args.al_cycle.n_init)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model_dict = build_model(args, n_classes=ds_info['n_classes'], train_ds=al_dataset.labeled_dataset)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, lr_scheduler = model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args)

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_dataset=al_dataset,
                acq_size=args.al_cycle.acq_size,
                batch_size=args.val_batch_size,
                device=args.device
            )
            al_dataset.update_annotations(indices)
            query_time = time.time() - t1
            logging.info('Querying took %.2f minutes', query_time/60)
            cycle_results['query_time'] = query_time

        #  If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_model_state)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        t1 = time.time()
        train_history = []
        for i_epoch in range(args.model.n_epochs):
            train_loader = DataLoader(al_dataset.labeled_dataset,
                                      batch_size=args.model.batch_size, shuffle=True, drop_last=True)
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


def build_query(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    elif args.al_strategy.name == "uncertainty":
        al_kwargs = {
            'model': None,
            'al_dataset': None,
            'batch_size': None,  # TODO to init method
            'acq_size': None,  # TODO to init method
        }
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


if __name__ == "__main__":
    main()
