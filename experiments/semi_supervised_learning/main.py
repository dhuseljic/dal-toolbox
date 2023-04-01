import os
import json
import logging
import torch
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import destroy_process_group
from omegaconf import OmegaConf

from dal_toolbox.datasets import build_ssl_dataset
from dal_toolbox.datasets.samplers import DistributedSampler
from dal_toolbox.models import build_ssl_model
from dal_toolbox.models.utils.ssl_utils import FlexMatchThresholdingHook
from dal_toolbox.utils import seed_everything, init_distributed_mode


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    # Check and initialize ddp if possible
    use_distributed = init_distributed_mode(args)
    if use_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        args.device = f'cuda:{rank}'

    # Initial Setup (Seed, create output folder, SummaryWriter and results-container init)
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    misc = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets. Creating labeled pool with %s samples and \
        unlabeled pool with %s samples.', args.n_labeled_samples, args.n_unlabeled_samples)
    lb_ds, ulb_ds_weak, ulb_ds_strong, val_ds, ds_info = build_ssl_dataset(args)

    # TODO: Create some sort of build_dataloader method?
    
    # Setup samplers and dataloaders
    Sampler = DistributedSampler if use_distributed else RandomSampler
    n_iter_per_epoch = args.model.n_iter // args.model.n_epochs

    supervised_sampler = Sampler(lb_ds, num_samples=(n_iter_per_epoch * args.model.batch_size))
    random_sampler_weak_1 = Sampler(ulb_ds_weak, num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                          args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    random_sampler_weak_2 = Sampler(ulb_ds_weak, num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                        args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    random_sampler_strong = Sampler(ulb_ds_strong, num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                          args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    random_sampler_idx = Sampler(range(len(ulb_ds_weak)), num_samples=int(n_iter_per_epoch * args.model.batch_size *
                                          args.ssl_algorithm.u_ratio), generator=torch.Generator().manual_seed(args.random_seed))
    
    supervised_loader = DataLoader(lb_ds, batch_size=args.model.batch_size, 
                                   sampler=supervised_sampler)
    unsupervised_loader_weak_1 = DataLoader(ulb_ds_weak, batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_1)
    unsupervised_loader_weak_2 = DataLoader(ulb_ds_weak, batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_weak_2)
    unsupervised_loader_strong = DataLoader(ulb_ds_weak, batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_strong)
    unsupervised_loader_idx = DataLoader(range(len(ulb_ds_weak)), batch_size=int(
        args.model.batch_size*args.ssl_algorithm.u_ratio), sampler=random_sampler_idx)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model_dict = build_ssl_model(args, n_classes=ds_info['n_classes'])
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    lr_scheduler = model_dict['lr_scheduler']
    model = DistributedDataParallel(model) if use_distributed else model

    # Adding necessary dataloaders as train kwargs
    if args.ssl_algorithm.name == 'fully_supervised':
        model_dict['train_kwargs']['dataloader'] = supervised_loader
    else:
        model_dict['train_kwargs']['labeled_loader'] = supervised_loader
        if args.ssl_algorithm.name == 'pseudo_labels':
            model_dict['train_kwargs']['unlabeled_loader'] = unsupervised_loader_weak_1
        elif args.ssl_algorithm.name == 'pi_model':
            model_dict['train_kwargs']['unlabeled_loader_weak_1'] = unsupervised_loader_weak_1
            model_dict['train_kwargs']['unlabeled_loader_weak_2'] = unsupervised_loader_weak_2
        elif args.ssl_algorithm.name == 'fixmatch':
            model_dict['train_kwargs']['unlabeled_loader_weak'] = unsupervised_loader_weak_1
            model_dict['train_kwargs']['unlabeled_loader_strong'] = unsupervised_loader_strong
        elif args.ssl_algorithm.name == 'flexmatch':
            model_dict['train_kwargs']['unlabeled_loader_weak'] = unsupervised_loader_weak_1
            model_dict['train_kwargs']['unlabeled_loader_strong'] = unsupervised_loader_strong
            model_dict['train_kwargs']['unlabeled_loader_indices'] = unsupervised_loader_idx
            model_dict['train_kwargs']['fmth'] = FlexMatchThresholdingHook(ulb_dest_len=len(ulb_ds_weak), num_classes=ds_info['n_classes'], thresh_warmup=True)
        else:
            assert True, 'No valid ssl_algorithm chosen!'
        

    # Training Process
    history_train, history_test = [], []
    for i_epoch in range(args.model.n_epochs):
        if use_distributed:
            for loader in [supervised_loader, unsupervised_loader_idx, unsupervised_loader_strong, unsupervised_loader_weak_1, unsupervised_loader_weak_2]:
                loader.sampler.set_epoch(i_epoch)

        # Train model for one epoch
        logging.info('Training epoch %s', i_epoch)
        train_stats = train_one_epoch(
            model, **model_dict['train_kwargs'], epoch=i_epoch
        )
        if lr_scheduler and args.ssl_algorithm.name == 'fully_supervised':
            lr_scheduler.step()
        for key, value in train_stats.items():
            writer.add_scalar(tag=f"train/{key}", scalar_value=value, global_step=i_epoch)
        logging.info('Training stats: %s', train_stats)

        # Evaluate model on test set
        logging.info('Evaluation epoch %s', i_epoch)
        test_stats = evaluate(model, val_loader, dataloaders_ood={}, **model_dict['eval_kwargs'])
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test/{key}", scalar_value=value, global_step=i_epoch)
        logging.info('Evaluation stats: %s', test_stats)

        # Save results
        history_train.append(train_stats)
        history_test.append(test_stats)

    # Indices of torchvision dset are int64 which are not json compatible
    misc = {
        "labeled_indices": [int(i) for i in lb_ds.indices],
        "unlabeled_indices": [int(i) for i in ulb_ds_weak.indices]
    }

    results = {
        'train_history': history_train,
        'test_history': history_test,
        'misc': misc
    }

    fname = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", fname)
    with open(fname, 'w') as f:
        json.dump(results, f)

    if use_distributed:
        destroy_process_group()


if __name__ == "__main__":
    main()
