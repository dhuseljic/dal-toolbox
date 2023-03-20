import os
import json
import logging

import hydra
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from dal_toolbox.datasets import build_dataset, build_ood_datasets
from dal_toolbox.models import build_model
from dal_toolbox.utils import write_scalar_dict, seed_everything

from dal_toolbox.models.deterministic.trainer import BasicTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="uncertainty")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    writer = SummaryWriter(log_dir=args.output_dir)
    misc = {}

    # Load data
    train_ds, test_ds_id, ds_info = build_dataset(args)
    ood_datasets = build_ood_datasets(args, ds_info['mean'], ds_info['std'])
    if args.n_samples:
        logger.info('Creating random training subset with %s samples. Saving indices.', args.n_samples)
        indices_id = torch.randperm(len(train_ds))[:args.n_samples]
        train_ds = Subset(train_ds, indices=indices_id)
        misc['train_indices'] = indices_id.tolist()

    logger.info('Training on %s with %s samples.', args.dataset, len(train_ds))
    logger.info('Test in-distribution dataset %s has %s samples.', args.dataset, len(test_ds_id))
    for name, test_ds_ood in ood_datasets.items():
        logger.info('Test out-of-distribution dataset %s has %s samples.', name, len(test_ds_ood))

    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, shuffle=True, drop_last=True)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.test_batch_size)
    test_loaders_ood = {name: DataLoader(test_ds_ood, batch_size=args.test_batch_size)
                        for name, test_ds_ood in ood_datasets.items()}

    # Load model
    model_dict = build_model(args, n_samples=len(train_ds), n_classes=ds_info['n_classes'], train_ds=train_ds)
    model = model_dict['model']
    optimizer = model_dict['optimizer']
    criterion = model_dict['criterion']
    train_one_epoch = model_dict['train_one_epoch']
    evaluate = model_dict['evaluate']
    lr_scheduler = model_dict['lr_scheduler']

    trainer = BasicTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_one_epoch=train_one_epoch,
        evaluate=evaluate,
        lr_scheduler=lr_scheduler,
        output_dir=args.output_dir,
        summary_writer=writer,
        device=args.device
    )
    trainer.train(
        n_epochs=args.model.n_epochs,
        train_loader=train_loader,
        test_loaders={'test_loader_id': test_loader_id},
        eval_every=50,
        save_every=50
    )

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logger.info("Saving results to %s", fname)
    test_stats = trainer.evaluate(test_loader_id=test_loader_id, test_loader_ood=test_loaders_ood)
    results = {
        'test_stats': test_stats,
        'train_history': trainer.train_history,
        'test_history': trainer.test_history,
        'misc': misc
    }
    with open(fname, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
