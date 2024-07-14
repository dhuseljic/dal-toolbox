import os
import time
import json
import logging

import torch
import torch.nn as nn
import hydra
import math
import transformers

import lightning as L
import wandb

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dal_toolbox import datasets
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox import metrics
from dal_toolbox.models.utils.callbacks import MetricLogger
from utils import build_dataset, build_model, build_query, initialize_wandb, strategy_results
from lightning.pytorch.callbacks import LearningRateMonitor

@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    args.output_dir = os.path.expanduser(args.output_dir)
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    results = {}
    queried_indices = {}
    initialize_wandb(args)

    # Setup Data
    logging.info('Building dataset %s', args.dataset.name)
    data = build_dataset(args)
    collator = transformers.DataCollatorWithPadding(
        tokenizer=data.tokenizer,
        padding='longest',
        return_tensors='pt'
    )
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        train_batch_size=args.model.batch_size,
        predict_batch_size=args.model.batch_size*4,
        collator=collator
    )

    if args.al_cycle.init_pool_file is not None: 
        logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
        with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
            initial_indices = json.load(f)
        assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match'
        al_datamodule.update_annotations(initial_indices)
    else:
        logging.info('Creating random initial labeled pool with %s samples.', args.al_cycle.n_init)
        al_datamodule.random_init(n_samples=args.al_cycle.n_init)

    queried_indices['cycle0'] = sorted(al_datamodule.labeled_indices)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=data.num_classes, len_trainset=len(data.train_dataset))
    test_dataloader = DataLoader(
        data.test_dataset,
        batch_size=args.model.batch_size*4,
        shuffle=False,
        collate_fn=collator
    )
    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args)

    # DAL cycle
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        if i_acq != 0: 
            ts = time.time()
            logging.info('Querying %s samples with strategy %s', args.al_cycle.acq_size, args.al_strategy.name)
            logging.info('Subset size of the unlabeled pool is %s', args.dataset.train_subset)
            indices = al_strategy.query(
                model=model, 
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)
            
            query_time = time.time() - ts
            logging.info('Query Time: %f minutes', query_time/60)
            cycle_results['query_indices'] = sorted(indices)
            cycle_results['query_time'] = query_time
            queried_indices[f'cycle{i_acq}'] = sorted(indices)
        
        # Reset Parameters
        if args.al_cycle.cold_start:
            model.reset_states()

        # Overwrite scheduler with information about labeled instances
        model.lr_scheduler = transformers.get_linear_schedule_with_warmup(
             optimizer=model.optimizer,
             num_warmup_steps=math.ceil(args.model.n_epochs * len(al_datamodule.train_dataloader()) * args.model.optimizer.warmup_ratio),
             num_training_steps=args.model.n_epochs * len(al_datamodule.train_dataloader())
        )

        # Training
        trainer = L.Trainer(
            max_epochs=args.model.n_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[LearningRateMonitor(logging_interval='step')]
        )

        trainer.fit(model, al_datamodule)

        # Evaluation
        test_stats = {}
        predictions = trainer.predict(model, test_dataloader)

        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])

        test_stats['n_labeled_samples'] = len(al_datamodule.labeled_indices)
        test_stats = evaluate_cycle(logits, targets, args)

        cycle_results['test_stats'] = test_stats

        logging.info('[Cycle %s] test stats: %s', i_acq, test_stats)

        cycle_results.update({
            "labeled_indices": al_datamodule.labeled_indices,
            "n_labeled_samples": len(al_datamodule.labeled_indices),
            "unlabeled_indices": al_datamodule.unlabeled_indices,
            "n_unlabeled_indices": len(al_datamodule.unlabeled_indices)
        })

        results[f'cycle{i_acq}'] = cycle_results

        # wandb logging
        for key, value in test_stats.items():
            wandb.log(
                {key: value},
                step = len(al_datamodule.labeled_indices)
            )
    
    # Log Auc Results
    results = strategy_results(results)

    # Save Results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info('Saving results to %s', file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Save Indices 
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)
        
def evaluate_cycle(logits, targets, args):
    test_stats = {}

    if args.dataset.n_classes <= 2:
        test_f1_macro = metrics.f1_macro(logits, targets, 'binary')
        test_f1_micro = test_f1_macro
    else:
        test_f1_macro = metrics.f1_macro(logits, targets, 'macro')
        test_f1_micro = metrics.f1_macro(logits, targets, 'micro')
    
    test_stats.update({
        "test_acc": metrics.Accuracy()(logits, targets).item(),
        "test_f1_macro": test_f1_macro,
        "test_f1_micro": test_f1_micro,
        "test_acc_blc": metrics.balanced_acc(logits, targets)
    })
        
    return test_stats

if __name__ == '__main__':
    main()







def auc_results(logits, targets):
    pass
    



#%%

    # if args.al_cycle.init_pool_file is not None:
    #     logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
    #     with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
    #         initial_indices = json.load(f)
    #     assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
    #     al_dataset.update_annotations(initial_indices)
