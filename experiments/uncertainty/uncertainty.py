import os
import json
import logging

import hydra
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, RandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dal_toolbox.datasets import build_dataset, build_ood_datasets
from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.utils import seed_everything, init_distributed_mode


@hydra.main(version_base=None, config_path="./configs", config_name="uncertainty")
def main(args):
    use_distributed = init_distributed_mode(args)
    if use_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        args.device = f'cuda:{rank}'

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

    if use_distributed:
        train_sampler = DistributedSampler(train_ds)
    else:
        iter_per_epoch = len(train_ds) // args.model.batch_size + 1
        train_sampler = RandomSampler(train_ds, num_samples=(iter_per_epoch * args.model.batch_size))

    train_loader = DataLoader(train_ds, batch_size=args.model.batch_size, sampler=train_sampler)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.test_batch_size)
    test_loaders_ood = {name: DataLoader(test_ds_ood, batch_size=args.test_batch_size)
                        for name, test_ds_ood in ood_datasets.items()}

    # Load model
    model_dict = build_model(args, n_samples=len(train_ds), n_classes=ds_info['n_classes'], train_ds=train_ds)
    model = model_dict['model']
    optimizer = model_dict['optimizer']
    criterion = model_dict['criterion']
    lr_scheduler = model_dict['lr_scheduler']
    Trainer = model_dict['trainer']

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        output_dir=args.output_dir,
        summary_writer=writer,
        device=args.device,
        use_distributed=use_distributed,
    )
    trainer.train(
        n_epochs=args.model.n_epochs,
        train_loader=train_loader,
        test_loaders={'test_loader_id': test_loader_id},
        eval_every=args.eval_interval,
        save_every=args.eval_interval,
    )
    test_stats = trainer.evaluate(dataloader=test_loader_id, dataloaders_ood=test_loaders_ood)
    logger.info("Final test results: %s", test_stats)

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    logger.info("Saving results to %s", fname)
    results = {
        'test_stats': test_stats,
        'train_history': trainer.train_history,
        'test_history': trainer.test_history,
        'misc': misc
    }
    with open(fname, 'w') as f:
        json.dump(results, f)


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']

    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'lr_scheduler': lr_scheduler,
            'trainer': deterministic.trainer.DeterministicTrainer,
        }

    elif args.model.name == 'resnet18_labelsmoothing':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'lr_scheduler': lr_scheduler,
            'trainer': deterministic.trainer.DeterministicTrainer,
        }

    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(n_classes, args.model.n_passes, args.model.dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'lr_scheduler': lr_scheduler,
            'trainer': mc_dropout.trainer.MCDropoutTrainer,
        }

    elif args.model.name == 'resnet18_ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = deterministic.resnet.ResNet18(n_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
            )
            lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.model.n_epochs)
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = ensemble.voting_ensemble.Ensemble(members)
        criterion = nn.CrossEntropyLoss()
        optimizer = ensemble.voting_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = ensemble.voting_ensemble.EnsembleLRScheduler(lr_schedulers)
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'trainer': ensemble.trainer.EnsembleTrainer,
        }

    elif args.model.name == 'resnet18_sngp':
        model = sngp.resnet.resnet18_sngp(
            num_classes=10,
            input_shape=(3, 32, 32),
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.norm_bound,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=False,
            random_feature_type=args.model.gp.random_feature_type,
            scale_random_features=args.model.gp.scale_random_features,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'trainer': sngp.trainer.SNGPTrainer,
        }

    else:
        raise NotImplementedError(f'{args.model.name} not implemented')

    return model_dict


if __name__ == '__main__':
    main()
