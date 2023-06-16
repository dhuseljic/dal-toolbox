import os
import time
import json
import logging

import torch
import hydra

import lightning as L

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dal_toolbox import datasets
from dal_toolbox.models.deterministic import DeterministicModel, resnet
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import RandomSampling, EntropySampling, Badge, CoreSet
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox.metrics import Accuracy
from dal_toolbox.models.utils.callbacks import MetricLogger


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    queried_indices = {}

    # Setup Dataset
    logging.info('Building datasets..')
    if args.dataset == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)
    elif args.dataset == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)
    else:
        raise NotImplementedError(f'Experiment not implemented for {args.dataset}')
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    test_loader = DataLoader(data.test_dataset, batch_size=args.model.predict_batch_size)
    al_datamodule.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_datamodule.labeled_indices

    logging.info('Building query strategy: %s', args.al_strategy.name)
    if args.al_strategy.name == "random":
        al_strategy = RandomSampling()
    elif args.al_strategy.name == "entropy":
        al_strategy = EntropySampling(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "coreset":
        al_strategy = CoreSet(subset_size=args.al_strategy.subset_size)
    elif args.al_strategy.name == "badge":
        al_strategy = Badge(subset_size=args.al_strategy.subset_size)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")

    # Setup Model
    logging.info('Building model..')
    model = resnet.ResNet18(num_classes=data.num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model.optimizer.lr,
        weight_decay=args.model.optimizer.weight_decay,
        momentum=args.model.optimizer.momentum
    )
    lr_scheduler = CosineAnnealingLRLinearWarmup(optimizer, num_epochs=args.model.num_epochs, warmup_epochs=10)
    model = DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': Accuracy()},
        val_metrics={'val_acc': Accuracy()},
    )

    # Active Learning Cycle
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size,
            )
            al_datamodule.update_annotations(indices)
            query_time = time.time() - t1
            logging.info('Querying took %.2f minutes', query_time/60)
            cycle_results['query_indices'] = indices
            cycle_results['query_time'] = query_time
            queried_indices[f'cycle{i_acq}'] = indices

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_datamodule.labeled_indices))
        model.reset_states()
        callbacks = []
        if is_running_on_slurm():
            callbacks.append(MetricLogger())
        trainer = L.Trainer(
            max_epochs=args.model.num_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            enable_progress_bar=(is_running_on_slurm() is False),
            default_root_dir=args.output_dir,
            check_val_every_n_epoch=args.val_interval,
            logger=False,
        )
        trainer.fit(model, al_datamodule)

        test_stats = {}
        acc_fn = Accuracy()
        predictions = trainer.predict(model, test_loader)
        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])
        test_stats['num_labeled_samples'] = len(al_datamodule.labeled_indices)
        test_stats['accuracy'] = acc_fn(logits, targets).item()
        cycle_results['test_stats'] = test_stats
        logging.info('Cycle test stats: %s', test_stats)

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

    # Save indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)


if __name__ == "__main__":
    main()
