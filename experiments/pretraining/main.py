import os
import time
import json
import logging

import hydra
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import destroy_process_group
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_dataset
from dal_toolbox.models.deterministic import wide_resnet
from dal_toolbox.models.deterministic.train import train_one_epoch
from dal_toolbox.models.deterministic.evaluate import evaluate
from dal_toolbox.utils import seed_everything, init_distributed_mode


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
    model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.model.optimizer.lr,
        weight_decay=args.model.optimizer.weight_decay,
        momentum=args.model.optimizer.momentum,
        nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)

    if use_distributed:
        model = DistributedDataParallel(model, device_ids=[rank])

    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        if use_distributed:
            train_sampler.set_epoch(i_epoch)
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, args.device, epoch=i_epoch)
        logging.info("Epoch [%s] - End of Training. Results: %s", i_epoch, train_stats)
        if lr_scheduler:
            lr_scheduler.step()
        history_train.append(train_stats)

        # Eval
        if (i_epoch+1) % args.eval_interval == 0 or (i_epoch+1) == args.model.n_epochs:
            logging.info("Epoch [%s]  - Start of Evaluation.", i_epoch)
            test_stats = evaluate(model.module, test_loader, {}, criterion, args.device)
            history_test.append(test_stats)
            logging.info("Epoch [%s] - End of Evaluation. Results: %s", i_epoch, test_stats)

            # Saving checkpoint
            t1 = time.time()
            logging.info('Saving checkpoint')
            checkpoint = {
                "args": args,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i_epoch,
                "train_history": history_train,
                "test_history": history_test,
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            logging.info('Saving took %.2f minutes', (time.time() - t1)/60)

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

    if use_distributed:
        destroy_process_group()


if __name__ == "__main__":
    main()
