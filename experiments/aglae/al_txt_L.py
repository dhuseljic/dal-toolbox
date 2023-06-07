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

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dal_toolbox import datasets
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox import metrics
from dal_toolbox.models.utils.callbacks import MetricLogger
from utils import build_dataset, build_model, build_query, initialize_wandb

@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_path, exist_ok=True)

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

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args)

    # DAL cycle
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        if i_acq != 0: 
            logging.info('Querying %s samples with strategy %s', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model, 
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)
            break

if __name__ == '__main__':
    main()
#%%

    # if args.al_cycle.init_pool_file is not None:
    #     logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
    #     with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
    #         initial_indices = json.load(f)
    #     assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
    #     al_dataset.update_annotations(initial_indices)
