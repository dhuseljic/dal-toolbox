import os
import json
import logging
import random
import hydra

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import lightning as L

from omegaconf import OmegaConf

from dal_toolbox.utils import seed_everything
from dal_toolbox.models.deterministic import resnet, wide_resnet
from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.callbacks import MetricLogger, MetricHistory
from dal_toolbox.datasets import CIFAR10, CIFAR100, SVHN


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
    datamodule = build_finetune_datamodule(data_dir=args.data_dir, dataset=args.dataset)
    if args.subset_size:
        train_indices = random.sample(range(len(datamodule.train_dataset)), k=args.subset_size)
        train_loader = DataLoader(Subset(datamodule.train_dataset, train_indices), batch_size=args.model.train_batch_size, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(datamodule.train_dataset, train_indices, batch_size=args.model.train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(datamodule.test_dataset, batch_size=args.model.test_batch_size)

    # Setup Model and apply changes according to SIMCLR
    logging.info('Building model.')
    model = build_model(args, num_classes=datamodule.num_classes)
    
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
    trainer.fit(model=model, train_dataloaders=train_loader)
    
    # Evaluate the finetuned model
    logging.info(f"Evaluating the finetuned model.")
    test_predictions = trainer.predict(model, test_loader)
    logits = torch.cat([pred[0] for pred in test_predictions])
    targets = torch.cat([pred[1] for pred in test_predictions])
    test_accuracy = (torch.sum(logits.softmax(-1).argmax(-1) == targets)/targets.shape[0]).item() * 100
    logging.info(f"The {args.model.name} reached a test accuracy of {round(test_accuracy, 2)}% on {args.dataset}!")

    # Saving results
    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    results = {'train_metrics':metrics_history.metrics, 'test_accuracy':test_accuracy}

    with open(fname, 'w') as f:
        json.dump(results, f)


def build_model(args, num_classes):
    # Select a model
    if args.model.name == 'resnet18':
        model = resnet.ResNet18(num_classes=num_classes)
    elif args.model.name == 'wideresnet282':
        model = wide_resnet.wide_resnet_28_2(num_classes=num_classes, dropout_rate=0.0)
    elif args.model.name == 'wideresnet2810':
        model = wide_resnet.wide_resnet_28_10(num_classes=num_classes, dropout_rate=0.0)
    else:
        raise AssertionError(f'Model {args.model.name} not implemented.')
    
    # Load pretrained weights
    if args.load_pretrained_weights and os.path.exists(args.pretrained_weights_path):
        logging.info(f"Loading pretrained weights from {args.pretrained_weights_path}")
        input_dim = model.linear.in_features
        model.linear = nn.Identity()
        model.load_state_dict(torch.load(args.pretrained_weights_path))
        model.linear = nn.Linear(in_features=input_dim, out_features=num_classes)

    # Freeze all layers except the final linear one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(params=model.linear.parameters(), **args.model.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
    
    return deterministic.DeterministicModel(model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=optimizer, lr_scheduler=lr_scheduler)


def build_finetune_datamodule(data_dir='./data', dataset='CIFAR10', val_split=0.1):
    if dataset == 'CIFAR10':
        datamodule = CIFAR10(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'CIFAR100':
        datamodule = CIFAR100(dataset_path=data_dir, val_split=val_split)
    elif dataset == 'SVHN':
        datamodule = SVHN(dataset_path=data_dir, val_split=val_split)
    else:
        raise AssertionError('Dataset not implemented.')
    return datamodule
    

if __name__ == "__main__":
    main()