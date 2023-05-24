#%%
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
from dal_toolbox.active_learning.strategies import random, badge
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox.metrics import Accuracy
from dal_toolbox.models.utils.callbacks import MetricLogger

@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    queried_indices = {}

    logging.info('Building datasets...')
    data = datasets.AGNnews("test", 0.1)


#%%
# from dal_toolbox import datasets
from dal_toolbox import models
import numpy as np


#%%
data = datasets.agnews.AGNews(
    modelname="bert-base-cased",
    dataset_path="ag_news")


#%%
data.full_train_dataset

#%%
main()
#%%
if __name__ == "__main__":
    main()
