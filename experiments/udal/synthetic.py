import os
import time
import json
import logging

import torch
import torch.nn as nn
import hydra

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset
from omegaconf import OmegaConf

from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.utils import seed_everything
from dal_toolbox.active_learning.strategies import random, uncertainty, query
from dal_toolbox.metrics.ood import entropy_fn

from active_learning import build_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD
from dal_toolbox.models.deterministic.resnet import ResNet18
from dal_toolbox.models import ensemble
from dal_toolbox.metrics.ood import ensemble_entropy_from_logits
from dal_toolbox.metrics import generalization


def gt_proba_mapping(pixel_sum):
    val = np.cos(pixel_sum*2*np.pi)+1
    val = val / 2
    return val


@hydra.main(version_base=None, config_path="./configs", config_name="synthetic")
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
    al_dataset.random_init(n_samples=args.al_cycle.n_init, class_balanced=True)
    queried_indices['cycle0'] = al_dataset.labeled_indices

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    trainer = build_model(args, n_classes=2)

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args, device=args.device)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            t1 = time.time()
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)

            indices = al_strategy.query(
                model=trainer.model,
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

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        iter_per_epoch = len(al_dataset.labeled_dataset) // args.model.batch_size + 1
        train_sampler = RandomSampler(al_dataset.labeled_dataset, num_samples=args.model.batch_size*iter_per_epoch)
        train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, sampler=train_sampler)

        trainer.reset_states(reset_model=args.al_cycle.cold_start)
        history = trainer.train(args.model.n_epochs, train_loader)
        cycle_results['train_history'] = history['train_history']

        # Evaluate resulting model using the ground truth probabilities
        # test_stats = trainer.evaluate(val_loader)
        logging.info('Testing using the ground truth probabilties..')
        # Collect all predictions
        model = trainer.model
        logits_list = []
        gt_probas_list = []
        targets_list = []
        for inputs, targets in val_loader:
            with torch.no_grad():
                logits = model(inputs.to(args.device)).cpu()
            gt_probas = gt_proba_mapping(inputs.mean(dim=(1, 2, 3)))
            logits_list.append(logits)
            targets_list.append(targets)
            gt_probas_list.append(gt_probas)
        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)
        gt_probas = torch.cat(gt_probas_list)
        gt_probas = torch.stack((1-gt_probas, gt_probas), dim=1)

        # Note that TCE does not make sense as the brier just describes it without binning
        acc1, = generalization.accuracy(logits, targets)
        nll = nn.CrossEntropyLoss(reduction='mean')(logits, gt_probas)
        brier = nn.MSELoss(reduction='mean')(logits.softmax(-1), gt_probas)
        test_stats = dict(
            test_acc1=acc1.item(),
            test_nll=nll.item(),
            test_brier=brier.item(),
        )
        logging.info('Test stats %s', test_stats)
        cycle_results['test_stats'] = test_stats

        # Log stuff
        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })
        results[f'cycle{i_acq}'] = cycle_results

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


def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    elif args.al_strategy.name == "uncertainty":
        device = kwargs['device']
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type, device=device,)
    elif args.al_strategy.name == "aleatoric":
        query = Aleatoric()
    elif args.al_strategy.name == "epistemic":
        query = Epistemic(
            ensemble_size=20,
            num_epochs=20,
            lr=1e-2,
            momentum=0.9,
            weight_decay=0.01,
            device=kwargs['device'],
        )
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


class Aleatoric(query.Query):

    @torch.no_grad()
    def query(self, dataset, unlabeled_indices, acq_size, **kwargs):
        del kwargs

        gt_probas_pos = []
        for idx in unlabeled_indices:
            img, _ = dataset[idx]
            rel_pixel_sum = img.mean()
            gt_proba_pos = gt_proba_mapping(rel_pixel_sum)
            gt_probas_pos.append(gt_proba_pos)
        gt_probas_pos = torch.stack(gt_probas_pos)
        gt_probas = torch.stack((1-gt_probas_pos, gt_probas_pos), dim=1)
        scores = entropy_fn(gt_probas)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices


class Epistemic(query.Query):
    def __init__(self, ensemble_size, num_epochs, lr, weight_decay, momentum, device, random_seed=None):
        super().__init__(random_seed)
        self.num_epochs = num_epochs
        self.ensemble_size = ensemble_size
        self.num_classes = 2
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def build_ensemble(self):
        members = [ResNet18(num_classes=self.num_classes) for _ in range(self.ensemble_size)]
        optimizers = [SGD(mem.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                          momentum=self.momentum) for mem in members]
        lr_schedulers = [CosineAnnealingLR(opt, T_max=self.num_epochs) for opt in optimizers]

        model = ensemble.voting_ensemble.Ensemble(members)
        criterion = nn.CrossEntropyLoss()
        optimizer = ensemble.voting_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = ensemble.voting_ensemble.EnsembleLRScheduler(lr_schedulers)
        trainer = ensemble.trainer.EnsembleTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=self.device,
        )
        return trainer

    def query(self, dataset, unlabeled_indices, acq_size, **kwargs):
        del kwargs

        # Train ensemble
        trainer = self.build_ensemble()
        labeled_indices = [idx for idx in range(len(dataset)) if idx not in unlabeled_indices]
        train_loader = DataLoader(dataset, sampler=labeled_indices, batch_size=64)
        trainer.train(n_epochs=self.num_epochs, train_loader=train_loader)

        # Select with highest epistemic uncertainty
        unlabeled_loader = DataLoader(dataset, sampler=unlabeled_indices, batch_size=64)
        logits = []
        for inputs, _ in unlabeled_loader:
            with torch.no_grad():
                logits.append(trainer.model.forward_sample(inputs.to(self.device)).cpu())
        logits = torch.cat(logits)
        scores = ensemble_entropy_from_logits(logits)
        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]

        return actual_indices


if __name__ == "__main__":
    main()
