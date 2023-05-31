#%%
import os
import time
import json
import logging

import torch
import hydra
import wandb

import lightning as L

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dal_toolbox import datasets
from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.lr_scheduler import CosineAnnealingLRLinearWarmup
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import random, badge
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox import metrics
from dal_toolbox.models.utils.callbacks import MetricLogger

from dal_toolbox.datasets import activeglae

@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    queried_indices = {}
    initialize_wandb(args)

    logging.info('Building datasets...')
    data = build_dataset(args)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=data.full_train_dataset,
        query_dataset=data.query_dataset,
        val_dataset=data.val_dataset,
        train_batch_size=args.model.batch_size,
        predict_batch_size=args.model.batch_size*4
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

    queried_indices['cycle0'] = al_datamodule.labeled_indices

    #Setup Model
    logging.info('Building model..')

    model = models.deterministic.Bert





        


def initialize_wandb(args):
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        reinit=args.wandb.reinit,
        #mode=args.wandb.mode,
        mode = 'disabled',
        name=args.model.name+'_'+args.al_strategy.name+'_'+args.dataset.name+'#'+str(args.random_seed),
        config = OmegaConf.to_container(
            args, 
            resolve=True, 
            throw_on_missing=True
        )
    )


#%%
from dal_toolbox import datasets

#%%
from dal_toolbox.datasets import activeglae
#%%
import transformers
transformers.logging.set_verbosity_error()
data = activeglae.TREC6(
    model_name="bert-base-cased",
    dataset_path="~/hf_datasets",
    val_split=0.1)

#%%

import transformers
transformers.logging.set_verbosity_error()
data = activeglae.Yelp5(
    model_name="bert-base-cased",
    dataset_path="~/hf_datasets/manual_ds",
    val_split=0.1)

#%%
data.train_dataset[0]
#%%
test = data.full_train_dataset.train_test_split(0)
#%%
test["test"]
#%%
test
#%%
from datasets import load_dataset
dataset = load_dataset("ag_news")

#%%
dataset['train']['text'][0]

#%%

data.full_train_dataset
#%%
data.train_dataset[0]
#%%
loader = DataLoader(data.full_train_dataset, batch_size=8)

for i in loader:
    print(i)
    break





data.train_dataset[0]
#%%
ds = load_dataset
#%%
data = datasets.Banks77(
    modelname="bert-base-cased",
    dataset_path="banking77",
    val_split=0.1
)
#%%
len(data.train_dataset)
#%%
len(data.full_train_dataset)


#%%
data.full_train_dataset

#%%
main()
#%%
if __name__ == "__main__":
    main()

def build_dataset(args):
    if args.dataset.name == 'agnews':
        data = activeglae.AGNews(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )
    
    elif args.dataset.name == 'banks77':
        data = activeglae.Banks77(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'dbpedia':
        data = activeglae.DBPedia(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'fnc1':
        data = activeglae.FNC1(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'mnli':
        data = activeglae.MNLI(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'qnli':
        data = activeglae.QNLI(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'sst2':
        data = activeglae.SST2(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'trec6':
        data = activeglae.TREC6(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'wikitalk':
        data = activeglae.Wikitalk(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    elif args.dataset.name == 'yelp5':
        data = activeglae.Yelp5(
            model_name=args.model.name_hf,
            dataset_path=args.dataset.path,
            val_split=args.dataset.split
        )

    else
        raise NotImplementedError('Dataset not available')
    
    return data 

def build_model(args, **kwargs):
    num_classes = kwargs['n_classes']

    if args.model.name == 'bert':
        model = deterministic.bert.BertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )
        model = deterministic.DeterministicAGLAEModel(
            model=None,
            loss_fn=None, 
            optimizer=None,
            lr_scheduler=None,
            train_metrics={'train_acc': metrics.Accuracy()},
            val_metrics=None
        )
        
    elif args.model.name ==
#%%

    # if args.al_cycle.init_pool_file is not None:
    #     logging.info('Using initial labeled pool from %s.', args.al_cycle.init_pool_file)
    #     with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
    #         initial_indices = json.load(f)
    #     assert len(initial_indices) == args.al_cycle.n_init, 'Number of samples in initial pool file does not match.'
    #     al_dataset.update_annotations(initial_indices)
