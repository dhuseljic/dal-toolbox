import os
import json
import logging

import hydra
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning as L

from omegaconf import OmegaConf

from dal_toolbox.utils import seed_everything
from dal_toolbox.models.deterministic import resnet, wide_resnet
from dal_toolbox.models.utils.callbacks import MetricLogger, MetricHistory
from dal_toolbox.datasets import CIFAR10SimCLR, CIFAR100SimCLR, SVHNSimCLR

from simclr_utils import SimCLR

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Initial Setup (Create required paths, enable reproducability and print args in logging)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)

    # Setup Dataset
    logging.info('Building dataset.')
    datamodule = build_pretrain_datamodule(data_dir=args.data_dir, dataset=args.dataset)
    train_loader = DataLoader(datamodule.train_dataset, batch_size=args.model.train_batch_size)
    val_loader = DataLoader(datamodule.val_dataset, batch_size=args.model.train_batch_size)

    # Setup Model and apply changes according to SIMCLR
    logging.info('Building model.')
    model = build_model(args)
    
    # Pretrain the Model
    metrics_logger = MetricLogger(log_interval=50)
    metrics_history = MetricHistory()
    trainer = L.Trainer(
        max_epochs=args.model.n_epochs, 
        callbacks=[metrics_logger, metrics_history],
        default_root_dir=args.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator='gpu'
        )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Saving results
    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    results = metrics_history.metrics

    with open(fname, 'w') as f:
        json.dump(results, f)



def build_model(args):
    if args.model.name == 'resnet18':
        encoder = resnet.ResNet18(num_classes=1)
    elif args.model.name == 'wideresnet282':
        encoder = wide_resnet.wide_resnet_28_2(num_classes=1, dropout_rate=0.0)
    elif args.model.name == 'wideresnet2810':
        encoder = wide_resnet.wide_resnet_28_10(num_classes=1, dropout_rate=0.0)
    else:
        raise AssertionError(f'Model {args.model.name} not implemented.')
    
    input_dim = encoder.linear.in_features
    output_dim = args.model.projection_dim
    encoder.linear = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, output_dim)
    )
    
    model = SimCLR(
        encoder=encoder, 
        projector=projector, 
        temperature=args.model.temperature, 
        optimizer_args=args.model.optimizer,
        n_epochs=args.model.n_epochs,
        model_dir=args.model_dir
    )
    return model


def build_pretrain_datamodule(data_dir='./data', dataset='CIFAR10', val_split=0.1):
    if dataset == 'CIFAR10':
        datamodule = CIFAR10SimCLR(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'CIFAR100':
        datamodule = CIFAR100SimCLR(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'SVHN':
        datamodule = SVHNSimCLR(dataset_path=data_dir, val_split=val_split)
    else:
        raise AssertionError('Dataset not implemented.')
    return datamodule
    

if __name__ == "__main__":
    main()