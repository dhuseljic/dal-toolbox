import os
import copy
import json
import logging
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
    history = []
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets. Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
    train_ds, query_ds, val_ds, n_classes = build_al_datasets(args)
    al_dataset = ALDataset(train_ds, query_ds)
    al_dataset.random_init(n_samples=args.al_cycle.n_init)

    # Setup Model #TODO: Does DUE need labels? Otherwise we can input the whole train ds instead of labeled samples
    logging.info('Building model: \n%s', OmegaConf.to_yaml(args.model))
    model_dict = build_model(args, n_classes=n_classes, train_ds=al_dataset.labeled_dataset)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, lr_scheduler = model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup Query
    logging.info('Building query strategy: \n%s', OmegaConf.to_yaml(args.al_strategy))
    al_strategy = build_query(args)

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            logging.info('Querying %s samples.', args.al_cycle.acq_size)
            indices = al_strategy.query(
                model=model,
                dataset=al_dataset,
                acq_size=args.al_cycle.acq_size,
                batch_size=args.val_batch_size,
                device=args.device
            )
            al_dataset.update_annotations(indices)

        #  If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_model_state)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        train_history = []
        for i_epoch in range(args.model.n_epochs):
            train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, shuffle=True, drop_last=True)
            train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
            if lr_scheduler:
                lr_scheduler.step()

            for key, value in train_stats.items():
                writer.add_scalar(tag=f"cycle_{i_acq}_train/{key}", scalar_value=value, global_step=i_epoch)
            train_history.append(train_stats)

        # Evaluate resulting model
        logging.info('Evaluation with %s samples', len(val_ds))
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=True)
        test_stats = evaluate(model, val_loader, dataloaders_ood={}, **model_dict['eval_kwargs'])
        logging.info(test_stats)

        # Log
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test_stats/{key}", scalar_value=value, global_step=i_acq)
        history.append({
            "train_history": train_history,
            "test_stats": test_stats,
            # TODO: add additional variables to test_stats
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })

    # Save results
    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    # torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
    with open(fname, 'w') as f:
        json.dump(history, f)


def build_query(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "uncertainty":
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


if __name__ == "__main__":
    main()
