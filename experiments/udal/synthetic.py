import os
import time
import json
import logging

import torch
import torch.nn as nn
import hydra

import numpy as np
import lightning as L

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, Subset
from omegaconf import OmegaConf

from dal_toolbox.active_learning.strategies import random, uncertainty, query
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.utils import seed_everything, is_running_on_slurm
from dal_toolbox.metrics import Accuracy, entropy_from_probas, ensemble_entropy_from_logits
from dal_toolbox.models.deterministic.resnet import ResNet18
from dal_toolbox.models.ensemble import EnsembleModel
from dal_toolbox.models.utils.callbacks import MetricLogger
from active_learning import build_model


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

    # Setup Dataset
    logging.info('Building datasets.')
    data_dict = torch.load(args.dataset_path)
    dataset = TensorDataset(data_dict['instances'], data_dict['targets'])
    n_train_samples = int(len(dataset)*.7)
    rnd_indices = torch.randperm(len(dataset))
    train_indices = rnd_indices[:n_train_samples]
    val_indices = rnd_indices[n_train_samples:]
    train_ds = Subset(dataset, indices=train_indices)
    test_ds = Subset(dataset, indices=val_indices)

    al_datamodule = ActiveLearningDataModule(train_dataset=train_ds)
    al_datamodule.random_init(n_samples=args.al_cycle.n_init, class_balanced=True)
    queried_indices['cycle0'] = al_datamodule.labeled_indices
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=2)

    # Setup Query
    logging.info('Building query strategy: %s', args.al_strategy.name)
    al_strategy = build_query(args)

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        cycle_results = {}

        # Analyse unlabeled set and query most promising data
        if i_acq != 0:
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al_cycle.acq_size
            )
            al_datamodule.update_annotations(indices)

        # Train with updated annotations
        logging.info('Training on labeled pool with %s samples', len(al_datamodule.labeled_indices))
        model.reset_states(reset_model_parameters=args.cold_start)
        trainer = L.Trainer(
            max_epochs=args.model.n_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=(not is_running_on_slurm()),
            callbacks=[MetricLogger()] if is_running_on_slurm() else [],
        )
        trainer.fit(model, al_datamodule)

        # Evaluate resulting model using the ground truth probabilities
        logging.info('Testing using the ground truth probabilties..')
        predictions = trainer.predict(model, test_loader)
        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])
        gt_probas = torch.cat([gt_proba_mapping(batch[0].mean(dim=(1, 2, 3))) for batch in test_loader])
        gt_probas = torch.stack((1-gt_probas, gt_probas), dim=1)

        # Note that TCE does not make sense as the brier just describes it without binning
        accuracy = Accuracy()(logits, targets)
        nll = nn.CrossEntropyLoss(reduction='mean')(logits, gt_probas)
        brier = nn.MSELoss(reduction='mean')(logits.softmax(-1), gt_probas)
        test_stats = dict(
            test_acc1=accuracy.item(),
            test_nll=nll.item(),
            test_brier=brier.item(),
        )
        logging.info('Test stats %s', test_stats)
        cycle_results['test_stats'] = test_stats

        # Log stuff
        cycle_results.update({
            "labeled_indices": al_datamodule.labeled_indices,
            "n_labeled_samples": len(al_datamodule.labeled_indices),
            "unlabeled_indices": al_datamodule.unlabeled_indices,
            "n_unlabeled_samples": len(al_datamodule.unlabeled_indices),
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
        query = random.RandomSampling()
    elif args.al_strategy.name == "entropy":
        query = uncertainty.EntropySampling()
    elif args.al_strategy.name == "aleatoric":
        query = Aleatoric()
    elif args.al_strategy.name == "epistemic":
        query = Epistemic(
            ensemble_size=20,
            num_epochs=20,
            lr=1e-2,
            momentum=0.9,
            weight_decay=0.01,
        )
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


class Aleatoric(query.Query):

    @torch.no_grad()
    def query(self, *, al_datamodule, acq_size, **kwargs):
        del kwargs
        unlabeled_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader()

        gt_probas_pos = torch.cat([gt_proba_mapping(batch[0].mean(dim=(1, 2, 3))) for batch in unlabeled_loader])
        gt_probas = torch.stack((1-gt_probas_pos, gt_probas_pos), dim=1)
        scores = entropy_from_probas(gt_probas)
        _, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices


class Epistemic(query.Query):
    def __init__(self, ensemble_size, num_epochs, lr, weight_decay, momentum, random_seed=None):
        super().__init__(random_seed)
        self.num_epochs = num_epochs
        self.ensemble_size = ensemble_size
        self.num_classes = 2
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def build_ensemble(self):
        members = [ResNet18(num_classes=self.num_classes) for _ in range(self.ensemble_size)]
        optimizers = [SGD(mem.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                          momentum=self.momentum) for mem in members]
        lr_schedulers = [CosineAnnealingLR(opt, T_max=self.num_epochs) for opt in optimizers]

        criterion = nn.CrossEntropyLoss()
        model = EnsembleModel(members, criterion, optimizers, lr_schedulers, {'train_acc': Accuracy()})
        return model

    def query(self, *, al_datamodule, acq_size, **kwargs):
        del kwargs

        # Train ensemble
        model = self.build_ensemble()
        sampler = torch.utils.data.SubsetRandomSampler(indices=al_datamodule.labeled_indices)
        train_loader = DataLoader(al_datamodule.train_dataset, sampler=sampler, batch_size=32)
        trainer = L.Trainer(
            max_epochs=self.num_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[],
        )
        trainer.fit(model, train_loader)
        # trainer.logged_metrics.keys()
        # [f"{k}: {m}" for k, m in trainer.logged_metrics.items() if 'train_acc' in k]

        # Select with highest epistemic uncertainty
        unlabeled_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader()
        predictions = trainer.predict(model, unlabeled_loader)
        logits = torch.cat([pred[0] for pred in predictions])
        scores = ensemble_entropy_from_logits(logits)
        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]

        return actual_indices


if __name__ == "__main__":
    main()
