import os
import json
import logging
import hydra
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_dataset
from dal_toolbox.models.deterministic import wide_resnet
from dal_toolbox.models.deterministic.train import train_one_epoch
from dal_toolbox.models.deterministic.evaluate import evaluate
from dal_toolbox.utils import seed_everything

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group


def main(rank, world_size):
    ddp_setup(rank, world_size)
    with hydra.initialize(version_base=None, config_path="./configs", job_name="pretraining"):
        args = hydra.compose(config_name="config")

    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Dataset
    logging.info('Building datasets.')
    train_ds, test_ds, ds_info = build_dataset(args)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = wide_resnet.WideResNet(28, args.model.width, dropout_rate=0, num_classes=ds_info['n_classes'])
    model = DistributedDataParallel(model.cuda())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model.optimizer.lr,
        weight_decay=args.model.optimizer.weight_decay,
        momentum=args.model.optimizer.momentum,
        nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
    device = 'cuda'

    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        train_sampler.set_epoch(i_epoch)
        logging.info("Epoch [%s] - Start of Training.", i_epoch)
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=i_epoch)
        logging.info("Epoch [%s] - End of Training. Results: %s", i_epoch, train_stats)
        if lr_scheduler:
            lr_scheduler.step()
        history_train.append(train_stats)

        # Eval
        if rank == 0:
            if (i_epoch+1) % args.eval_interval == 0 or (i_epoch+1) == args.model.n_epochs:
                logging.info("Epoch [%s]  - Start of Evaluation.", i_epoch)
                test_stats = evaluate(model, test_loader, {}, criterion, device)
                history_test.append(test_stats)
                logging.info("Epoch [%s] - End of Evaluation. Results: %s", i_epoch, test_stats)

                # Saving checkpoint
                t1 = time.time()
                logging.info('Saving checkpoint')
                checkpoint = {
                    "args": args,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": i_epoch,
                    "train_history": history_train,
                    "test_history": history_test,
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
                logging.info('Saving took %.2f minutes', (time.time() - t1)/60)

    if rank == 0:
        # Saving results
        fname = os.path.join(args.output_dir, 'results_final.json')
        logging.info("Saving results to %s.", fname)
        torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
        results = {
            'train_history': history_train,
            'test_history': history_test
        }
        with open(fname, 'w') as f:
            json.dump(results, f)
    destroy_process_group()


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
