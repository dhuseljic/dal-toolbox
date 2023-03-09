import os
import time
import copy
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset
from omegaconf import OmegaConf

from dal_toolbox.models import deterministic, mc_dropout, ensemble, sngp
from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.utils import seed_everything
from dal_toolbox.active_learning.strategies import random, uncertainty, query
from dal_toolbox.metrics.ood import entropy_fn


@hydra.main(version_base=None, config_path="./configs", config_name="aleatoric")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    results = {}
    queried_indices = {}
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    logging.info('Building datasets.')
    data_dict = torch.load(args.dataset_path)
    dataset = TensorDataset(data_dict['instances'], data_dict['targets'])
    n_train_samples = int(len(dataset)*.7)
    rnd_indices = torch.randperm(len(dataset))
    train_indices = rnd_indices[:n_train_samples]
    val_indices = rnd_indices[n_train_samples:]
    train_ds = Subset(dataset, indices=train_indices)
    val_ds = Subset(dataset, indices=val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    al_dataset = ALDataset(train_ds, random_state=args.random_seed)
    al_dataset.random_init(n_samples=args.al_cycle.n_init)
    queried_indices['cycle0'] = al_dataset.labeled_indices

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model_dict = build_model(args, n_classes=2)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    criterion, optimizer, lr_scheduler = model_dict['criterion'], model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args, device=args.device)

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)

            indices = al_strategy.query(
                model=model,
                dataset=al_dataset.query_dataset,
                unlabeled_indices=al_dataset.unlabeled_indices,
                labeled_indices=al_dataset.labeled_indices,
                acq_size=args.al_cycle.acq_size
            )
            al_dataset.update_annotations(indices)

            query_time = time.time() - t1
            logging.info('Querying took %.2f minutes', query_time/60)
            cycle_results['query_indices'] = indices
            cycle_results['query_time'] = query_time
            queried_indices[f'cycle{i_acq}'] = indices

        #  If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_model_state)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        logging.info('Distribution of labels: %s ', np.unique([y for _, y in al_dataset.labeled_dataset], return_counts=True))
        t1 = time.time()
        iter_per_epoch = len(al_dataset.labeled_dataset) // args.model.batch_size + 1
        train_sampler = RandomSampler(al_dataset.labeled_dataset, num_samples=args.model.batch_size*iter_per_epoch)
        train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, sampler=train_sampler)
        train_history = []

        for i_epoch in range(args.model.n_epochs):
            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=i_epoch,
                **model_dict['train_kwargs'],
            )
            if lr_scheduler:
                lr_scheduler.step()

            for key, value in train_stats.items():
                writer.add_scalar(tag=f"cycle_{i_acq}_train/{key}", scalar_value=value, global_step=i_epoch)
            train_history.append(train_stats)
        training_time = (time.time() - t1)
        logging.info('Training took %.2f minutes', training_time/60)
        logging.info('Training stats: %s', train_stats)
        cycle_results['train_history'] = train_history
        cycle_results['training_time'] = training_time

        # Evaluate resulting model
        logging.info('Evaluation with %s samples', len(val_ds))
        t1 = time.time()
        test_stats = evaluate(model, val_loader, criterion=criterion,  dataloaders_ood={}, **model_dict['eval_kwargs'])
        evaluation_time = time.time() - t1
        logging.info('Evaluation took %.2f minutes', evaluation_time/60)
        logging.info('Evaluation stats: %s', test_stats)
        cycle_results['evaluation_time'] = evaluation_time
        cycle_results['test_stats'] = test_stats

        # Log
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test_stats/{key}", scalar_value=value, global_step=i_acq)

        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })
        results[f'cycle{i_acq}'] = cycle_results

        # Save checkpoint
        logging.info('Saving checkpoint for cycle %s', i_acq)
        checkpoint = {
            "args": args,
            "model": model.state_dict(),
            "al_dataset": al_dataset.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "cycle_results": cycle_results,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Saving
    # Save results
    file_name = os.path.join(args.output_dir, 'results.json')
    logging.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Save indices
    file_name = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info("Saving queried indices to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(queried_indices, f, sort_keys=False)

    # Save Model
    file_name = os.path.join(args.output_dir, "model_final.pth")
    logging.info("Saving final model to %s.", file_name)
    torch.save(checkpoint, file_name)


def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    elif args.al_strategy.name == "uncertainty":
        device = kwargs['device']
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type, device=device,)
    elif args.al_strategy.name == "aleatoric":
        query = Aleatoric()
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


# Defines aleatoric sampling
class Aleatoric(query.Query):

    @torch.no_grad()
    def query(self, dataset, unlabeled_indices, acq_size, **kwargs):
        del kwargs

        gt_probas_pos = []
        for idx in unlabeled_indices:
            img, _ = dataset[idx]
            gt_probas_pos.append(img.mean())
        gt_probas_pos = torch.stack(gt_probas_pos)
        gt_probas = torch.stack((1-gt_probas_pos, gt_probas_pos), dim=1)
        scores = entropy_fn(gt_probas)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices


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
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'train_one_epoch': deterministic.train.train_one_epoch,
            'evaluate': deterministic.evaluate.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(device=args.device),
            'eval_kwargs': dict(device=args.device),
        }

    elif args.model.name == 'resnet18_labelsmoothing':
        model = deterministic.resnet.ResNet18(n_classes)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'train_one_epoch': deterministic.train.train_one_epoch,
            'evaluate': deterministic.evaluate.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(device=args.device),
            'eval_kwargs': dict(device=args.device),
        }

    elif args.model.name == 'resnet18_mixup':
        NotImplementedError()

    elif args.model.name == 'resnet18_mcdropout':
        model = mc_dropout.resnet.DropoutResNet18(n_classes, args.model.n_passes, args.model.dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'train_one_epoch': mc_dropout.train.train_one_epoch,
            'evaluate': mc_dropout.evaluate.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(device=args.device),
            'eval_kwargs': dict(device=args.device),
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
                nesterov=True
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
            'train_one_epoch': ensemble.train.train_one_epoch,
            'evaluate': ensemble.evaluate.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(device=args.device),
            'eval_kwargs': dict(device=args.device),
        }

    elif args.model.name == 'resnet18_sngp':
        model = sngp.resnet.resnet18_sngp(
            num_classes=n_classes,
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
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        model_dict = {
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'train_one_epoch': sngp.train.train_one_epoch,
            'evaluate': sngp.evaluate.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(device=args.device),
            'eval_kwargs': dict(device=args.device),
        }

    else:
        NotImplementedError()

    return model_dict


if __name__ == "__main__":
    main()
