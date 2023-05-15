import os
import time
import json
import logging

import hydra
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from torch.distributed import destroy_process_group
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_dataset
from dal_toolbox.models.deterministic import wide_resnet
from dal_toolbox.models.deterministic.train import train_one_epoch
from dal_toolbox.models.deterministic.evaluate import evaluate
from dal_toolbox.utils import seed_everything, init_distributed_mode
from dal_toolbox.models.deterministic.trainer import DeterministicTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    use_distributed = init_distributed_mode(args)
    if use_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        args.device = f'cuda:{rank}'

    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)

    # Setup Dataset
    logging.info('Building datasets.')
    train_ds, test_ds, ds_info = build_dataset(args)

    if use_distributed:
        train_sampler = DistributedSampler(train_ds)
    else:
        train_sampler = RandomSampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = wide_resnet.WideResNet(28, args.model.width, dropout_rate=0, num_classes=ds_info['n_classes'])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model.optimizer.lr,
        weight_decay=args.model.optimizer.weight_decay,
        momentum=args.model.optimizer.momentum,
        nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)

    trainer = DeterministicTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        device=args.device,
        use_distributed=use_distributed,
    )
    trainer.train(args.model.n_epochs, train_loader=train_loader)
    test_stats = trainer.evaluate(test_loader_id=test_loader)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logging.info("Saving results to %s.", fname)
    results = {
        'train_history': trainer.train_history,
        'test_history': trainer.val_history,
        'test_stats': test_stats
    }
    with open(fname, 'w') as f:
        json.dump(results, f)

    if use_distributed:
        destroy_process_group()


if __name__ == "__main__":
    main()
