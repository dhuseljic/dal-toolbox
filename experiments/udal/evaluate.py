"""This script evaluates an active learning cycle with indices that are given as arguments."""
import os
import json
import logging

import torch
import hydra
import lightning as L

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox.models.utils.callbacks import MetricLogger
from active_learning import build_model, build_datasets, build_ood_datasets, evaluate, evaluate_ood


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}

    # Setup Dataset
    logging.info('Building datasets.')
    data = build_datasets(args)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        val_dataset=data.val_dataset,
        query_dataset=data.query_dataset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    test_loader = DataLoader(data.test_dataset, batch_size=args.model.predict_batch_size)
    if args.ood_datasets:
        logging.info('Building ood datasets.')
        ood_datasets = build_ood_datasets(args, id_mean=data.mean, id_std=data.std)
        ood_loaders = {name: DataLoader(ds, batch_size=args.model.predict_batch_size) for name, ds in ood_datasets.items()}
    else:
        ood_loaders = None

    logging.info('Loading initial labeled pool from %s', args.queried_indices_json)
    with open(args.queried_indices_json, 'r', encoding='utf-8') as f:
        queried_indices = json.load(f)
    al_datamodule.update_annotations(queried_indices['cycle0'])
    n_init = len(queried_indices['cycle0'])
    acq_size = len(queried_indices['cycle1'])
    n_acq = len(queried_indices) - 1
    logging.info('Initial pool: %s  Acquisition Size: %s  Number of Acquisitions: %s', n_init, acq_size, n_acq)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes)

    # Active Learning Cycles
    for i_acq in range(0, n_acq+1):
        logging.info('Starting AL iteration %s / %s', i_acq, n_acq)
        cycle_results = {}

        if i_acq != 0:
            indices = queried_indices[f'cycle{i_acq}']
            logging.info('Updating pool with %s samples', len(indices))
            al_datamodule.update_annotations(indices)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_datamodule.labeled_indices))
        model.reset_states(reset_model_parameters=args.cold_start)
        trainer = L.Trainer(
            max_epochs=args.model.n_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            logger=False,
            check_val_every_n_epoch=args.val_every,
            enable_progress_bar=(not is_running_on_slurm()),
            callbacks=[MetricLogger()] if is_running_on_slurm() else [],
            fast_dev_run=args.fast_dev_run
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model
        predictions = trainer.predict(model, test_loader)
        logits = torch.cat([preds[0] for preds in predictions])
        targets = torch.cat([preds[1] for preds in predictions])
        test_stats = evaluate(logits, targets)
        for name, loader in ood_loaders.items():
            predictions_ood = trainer.predict(model, loader)
            logits_ood = torch.cat([preds[0] for preds in predictions_ood])
            ood_stats = evaluate_ood(logits, logits_ood)
            ood_stats = {f'{key}_{name}': val for key, val in ood_stats.items()}
            test_stats.update(ood_stats)
        logging.info("[Acq %s] Test statistics: %s", i_acq, test_stats)

        cycle_results['test_stats'] = test_stats
        cycle_results.update({
            "labeled_indices": al_datamodule.labeled_indices,
            "n_labeled_samples": len(al_datamodule.labeled_indices),
            "unlabeled_indices": al_datamodule.unlabeled_indices,
            "n_unlabeled_samples": len(al_datamodule.unlabeled_indices),
        })
        results[f'cycle{i_acq}'] = cycle_results

    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
