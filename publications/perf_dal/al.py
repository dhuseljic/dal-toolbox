import time
import hydra
import torch
import mlflow
import copy
import logging
import itertools

import numpy as np
from lightning import Trainer
from omegaconf import OmegaConf

from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning.strategies.query import Query
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything
from utils import build_datasets, flatten_cfg, build_model
from torch.utils.data import DataLoader, Subset, ConcatDataset
from rich.progress import track

mlflow.config.enable_async_logging()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))
    train_ds, test_ds, num_classes = build_datasets(
        args, val_split=args.use_val_split, cache_features=args.cache_features)

    seed_everything(args.random_seed)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    al_strategy = build_al_strategy(args)
    # TODO: init should be part of al_strategy
    num_init = args.al.acq_size if args.al.num_init_samples is None else args.al.num_init_samples
    if args.al.init_method == 'random':
        al_datamodule.random_init(n_samples=num_init)
    else:
        raise NotImplementedError()

    num_features = len(train_ds[0][0])
    model = build_model(args, num_features=num_features, num_classes=num_classes)
    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        barebones=True,
        callbacks=[MetricLogger()],
    )

    al_history = []
    artifacts_history = []
    for i_acq in range(0, args.al.num_acq+1):
        if i_acq != 0:
            stime = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al.acq_size,
            )
            etime = time.time()
            al_datamodule.update_annotations(indices)

        artifacts = {
            'query_indices': indices if i_acq != 0 else al_datamodule.labeled_indices,
            'model': model.state_dict(),
        }

        model.reset_states()
        trainer = Trainer(**lightning_trainer_config)
        trainer.fit(model, train_dataloaders=al_datamodule.train_dataloader())

        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0
        if args.al.strategy == 'optimal':
            bought_dict = {f'bought_{k}': v for k, v in al_strategy.batch_type_count.items()}
            test_stats.update(bought_dict)

        print(f'Cycle {i_acq}:', test_stats, flush=True)
        al_history.append(test_stats)
        artifacts_history.append(artifacts)

    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(al_history):
        mlflow.log_metrics(test_stats, step=i_acq)
        mlflow.log_dict(artifacts_history[i_acq], f'artifacts_cycle{i_acq:02d}')
    mlflow.end_run()


def evaluate(predictions):
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_labels).item(),
        'NLL': metrics.CrossEntropy()(test_logits, test_labels).item(),
        'BS': metrics.BrierScore()(test_logits, test_labels).item(),
        'ECE': metrics.ExpectedCalibrationError()(test_logits, test_labels).item(),
        'ACE': metrics.AdaptiveCalibrationError()(test_logits, test_labels).item(),
        'reliability': metrics.BrierScoreDecomposition()(test_logits, test_labels)['reliability']
    }
    return test_stats


def build_al_strategy(args):
    device = args.al.device
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'optimal':
        al_strategy = Optimal(
            subset_size=args.al.subset_size,
            num_batches=args.al.optimal.num_batches,
            batch_types=args.al.optimal.batch_types,
            look_ahead=args.al.optimal.look_ahead,
            num_mc_labels=args.al.optimal.num_mc_labels,
            use_val_ds=args.al.optimal.use_val_ds,
            loss=args.al.optimal.loss,
            use_retraining=args.al.optimal.use_retraining,
            num_retraining_epochs=args.al.optimal.num_retraining_epochs,
            gamma=args.al.optimal.gamma,
            device=device
        )
    elif args.al.strategy == 'margin':
        al_strategy = strategies.MarginSampling(subset_size=args.al.subset_size, device=device)
    elif args.al.strategy == 'typiclust':
        al_strategy = strategies.TypiClust(subset_size=args.al.subset_size)
    elif args.al.strategy == 'badge':
        al_strategy = strategies.Badge(subset_size=args.al.subset_size)
    else:
        raise NotImplementedError()
    return al_strategy


class Optimal(Query):
    def __init__(self,
                 subset_size=None,
                 num_batches=200,
                 batch_types=['random', 'diverse', 'uncertain'],
                 num_mc_labels=None,
                 look_ahead='true_labels',
                 use_retraining=True,
                 use_val_ds=True,
                 num_retraining_epochs=10,
                 gamma=10,
                 loss='cross_entropy',
                 device='cpu',
                 random_seed=None,
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device

        # Batch Selection
        self.num_batches = num_batches
        self.batch_types = batch_types
        self.batch_type_count = {k: 0 for k in self.batch_types}

        # Look-Ahead - True labels, All comb, MC, Pseudo Labels
        self.look_ahead = look_ahead
        self.num_mc_labels = num_mc_labels

        # Performance Estimation
        self.use_val_ds = use_val_ds
        if loss == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss == 'expected_cross_entropy':
            self.loss_fn = lambda logits, _: torch.mean(
                torch.sum(logits.softmax(-1) * logits.log_softmax(-1), dim=-1))
        elif loss == 'zero_one':
            self.loss_fn = lambda logits, y: 1 - torch.mean((logits.argmax(dim=-1) == y).float())
        elif loss == 'expected_zero_one':
            self.loss_fn = lambda logits, _: torch.mean(1 - logits.softmax(-1).max(-1).values)
        else:
            raise NotImplementedError()

        # Retraining
        self.use_retraining = use_retraining
        self.num_retraining_epochs = num_retraining_epochs
        self.gamma = gamma

    def select_batches(self, acq_size, unlabeled_features, unlabeled_logits, labeled_features, labeled_labels, batch_types):
        # TODO: maybe focus more often on clusters with many samples? class distribution
        num_batch_types = len(batch_types)
        num_batches = self.num_batches // num_batch_types

        indices_batches = []

        # If there is a rest, add random batches
        num_rest_batches = self.num_batches % num_batch_types
        indices = [np.random.permutation(len(unlabeled_features))[:acq_size] for _ in range(num_rest_batches)]
        indices_batches.extend(indices)

        for batch_type in batch_types:
            if batch_type == 'random':
                indices = [np.random.permutation(len(unlabeled_features))[:acq_size]
                           for _ in range(num_batches)]
                indices_batches.extend(indices)

            elif batch_type == 'diverse': # KMeans with covered and uncovered clusters
                from sklearn.cluster import KMeans
                num_clusters = acq_size + len(labeled_features)
                km = KMeans(n_clusters=num_clusters, n_init='auto')
                features = torch.cat((labeled_features, unlabeled_features))
                clusters = km.fit_predict(features)
                cluster_sizes = np.zeros(num_clusters)
                cluster_ids, cluster_id_sizes = np.unique(clusters, return_counts=True)
                cluster_sizes[cluster_ids] = cluster_id_sizes
                covered_clusters = np.unique(clusters[:len(labeled_features)])
                if len(covered_clusters) > 0:
                    cluster_sizes[covered_clusters] = 0
                unlabeled_clusters = clusters[len(labeled_features):]

                # Random sample from uncovered clusters
                indices = []
                for _ in range(num_batches):
                    idx = []
                    cluster_sizes_batch = cluster_sizes.copy()
                    for _ in range(acq_size):
                        if np.any(cluster_sizes_batch != 0):
                            cluster_id = cluster_sizes_batch.argmax()
                            cluster_indices = (unlabeled_clusters == cluster_id).nonzero()[0]
                            idx.append(self.rng.choice(cluster_indices))
                            cluster_sizes_batch[cluster_id] = 0
                        else:
                            indices_ = np.arange(len(unlabeled_features))
                            indices_ = np.setdiff1d(indices_, idx)
                            idx.append(self.rng.choice(indices_))
                    indices.append(np.array(idx))
                indices_batches.extend(indices)

            elif batch_type == 'uncertain':
                probas = unlabeled_logits.softmax(-1)
                top_probas, _ = torch.topk(probas, k=2, dim=-1)
                margin_uncertainty = 1 - (top_probas[:, 0] - top_probas[:, 1])
                indices_uncertain = np.where(margin_uncertainty > margin_uncertainty.median())[0]
                indices = [self.rng.choice(indices_uncertain, size=acq_size, replace=False)
                           for _ in range(num_batches)]
                indices_batches.extend(indices)
            else:
                raise ValueError(f'Batch type {batch_type} not implemented')

        return indices_batches

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])
        unlabeled_logits = model.get_logits_from_representations(unlabeled_features, device=self.device)

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_features = model.get_representations(labeled_dataloader, device=self.device)
        labeled_labels = torch.cat([batch[1] for batch in labeled_dataloader])

        indices_batches = self.select_batches(
            acq_size,
            unlabeled_features,
            unlabeled_logits,
            labeled_features,
            labeled_labels,
            self.batch_types
        )

        loss_batches = []
        init_params = model.state_dict()
        for indices in track(indices_batches):
            if self.look_ahead == 'true_labels': # Use true labels of batch for model training
                labels = unlabeled_labels[indices]
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'pseudo_labels': # Use pseudo labels of model
                logits = unlabeled_logits[indices]
                labels = logits.argmax(-1)
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'mc_labels': # Samples labels via Monte Carlo
                categorical = torch.distributions.Categorical(logits=unlabeled_logits[indices])
                labels = categorical.sample((self.num_mc_labels,))
            elif self.look_ahead == 'all_labels': # Go through all label combinations
                num_classes = unlabeled_logits.size(-1)
                labels = itertools.product(range(num_classes), repeat=acq_size)
            else: 
                raise NotImplementedError()

            loss_labels = []
            for labels_batch in labels:

                if self.use_retraining:
                    model.reset_states(reset_model_parameters=True)
                    retrain_indices = labeled_indices + [unlabeled_indices[idx] for idx in indices]
                    custom_labels = torch.cat((labeled_labels, labels_batch))
                    retrain_loader = al_datamodule.custom_dataloader(
                        indices=retrain_indices, train=True, custom_labels=custom_labels)
                    trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
                    trainer.fit(model, retrain_loader)
                else:
                    model.load_state_dict(init_params)
                    retrain_indices = [unlabeled_indices[idx] for idx in indices]
                    retrain_loader = al_datamodule.custom_dataloader(
                        indices=retrain_indices, train=True, custom_labels=labels_batch)
                    model.update_posterior( 
                        retrain_loader,
                        gamma=self.gamma,
                        from_representations=True,
                        device=self.device
                    )

                if self.use_val_ds:
                    loss = self.evaluate_model(model, al_datamodule.val_dataloader())
                else:
                    loss = self.evaluate_model(model, al_datamodule.labeled_dataloader()[0])

                loss_labels.append(loss)
            loss_batches.append(np.mean(loss_labels))

        best_idx = np.argmin(loss_batches)
        
        num_batch_per_type = self.num_batches // len(self.batch_types)
        batch_type = self.batch_types[best_idx.item() // num_batch_per_type]
        self.batch_type_count[batch_type] += 1
        
        local_indices = indices_batches[best_idx]
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        return global_indices

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.to(self.device)
        num_samples = 0
        running_loss = 0
        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            logits = model(inputs)

            num_samples += len(inputs)
            running_loss += len(inputs)*self.loss_fn(logits, targets).item()
        loss = running_loss / num_samples
        return loss




if __name__ == '__main__':
    main()
