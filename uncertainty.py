import os
import logging
import json
import hydra
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataset, build_ood_datasets
from models import build_model
from utils import write_scalar_dict, seed_everything


@hydra.main(version_base=None, config_path="./configs", config_name="uncertainty")
def main(args):
    logging.basicConfig(filename=os.path.join(args.output_dir, 'uncertainty.log'), format='%(asctime)s %(message)s')
    logging.info(args)
    seed_everything(args.random_seed)
    writer = SummaryWriter(log_dir=args.output_dir)
    exp_info = {}

    # Load data
    train_ds, test_ds_id, ds_info = build_dataset(args)
    ood_datasets = build_ood_datasets(args, ds_info['mean'], ds_info['std'])
    if args.n_samples:
        logging.info(f'Creating random training subset with {args.n_samples} samples.')
        indices_id = torch.randperm(len(train_ds))[:args.n_samples]
        train_ds = Subset(train_ds, indices=indices_id)
        exp_info['train_indices'] =  indices_id.tolist()

    logging.info('Training on %s with %i samples.', args.dataset, len(train_ds))
    logging.info('Test in-distribution dataset %s has %i samples.', args.dataset, len(test_ds_id))
    for name, test_ds_ood in ood_datasets.items():
        logging.info(f'Test out-of-distribution dataset {name} has {len(test_ds_ood)} samples.')

    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, shuffle=True, drop_last=True)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.test_batch_size)
    test_loaders_ood = {name: DataLoader(test_ds_ood, batch_size=args.test_batch_size)
                        for name, test_ds_ood in ood_datasets.items()}

    # Load model
    model_dict = build_model(args, n_samples=len(train_ds), n_classes=ds_info['n_classes'], train_ds=train_ds)
    model = model_dict['model']
    train_one_epoch = model_dict['train_one_epoch']
    evaluate = model_dict['evaluate']
    lr_scheduler = model_dict['lr_scheduler']

    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
        if lr_scheduler:
            lr_scheduler.step()
        history_train.append(train_stats)
        write_scalar_dict(writer, prefix='train', dict=train_stats, global_step=i_epoch)

        # Eval
        if (i_epoch+1) % args.eval_interval == 0 or (i_epoch+1) == args.model.n_epochs:
            test_stats = evaluate(model, test_loader_id, test_loaders_ood, **model_dict['eval_kwargs'])
            history_test.append(test_stats)
            write_scalar_dict(writer, prefix='test', dict=test_stats, global_step=i_epoch)
            logging.info("Epoch [%i] %s %s", i_epoch, train_stats, test_stats)

        # Saving checkpoint
        checkpoint = {
            "args": args,
            "model": model.state_dict(),
            "optimizer": model_dict['train_kwargs']['optimizer'].state_dict(),
            "epoch": i_epoch,
            "train_history": history_train,
            "test_history": history_test,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
        }
        # torch.save(checkpoint, os.path.join(args.output_dir, f"model_{i_epoch}.pth"))
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logging.info("Saving results to %s.", fname)
    torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
    results = {
        'exp_info': exp_info,
        'train_history': history_train,
        'test_history': history_test,
    }
    with open(fname, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
