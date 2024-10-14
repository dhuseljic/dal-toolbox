import time
import hydra
import torch
import mlflow
import copy
import logging

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
    for i_acq in range(0, args.al.num_acq+1):
        if i_acq != 0:
            print('Querying..')
            stime = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al.acq_size,
            )
            etime = time.time()
            al_datamodule.update_annotations(indices)

        model.reset_states()
        trainer = Trainer(**lightning_trainer_config)
        trainer.fit(model, train_dataloaders=al_datamodule.train_dataloader())

        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0
        test_stats['query_indices'] = indices if i_acq != 0 else al_datamodule.labeled_indices
        if args.al.strategy == 'optimal':
            bought_dict = {f'bought_{k}': v for k, v in al_strategy.batch_type_count.items()}
            test_stats.update(bought_dict)
        print(f'Cycle {i_acq}:', test_stats, flush=True)
        al_history.append(test_stats)

    mlflow.set_tracking_uri(uri="{}".format(args.mlflow_uri))
    experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(al_history):
        mlflow.log_dict({'query_indices': test_stats.pop('query_indices')}, f'query_indices_{i_acq:02d}')
        mlflow.log_metrics(test_stats, step=i_acq, synchronous=False)
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
            use_true_labels=args.al.optimal.use_true_labels,
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
                 gamma=7,
                 num_mc_labels=None,
                 use_true_labels=True,
                 use_retraining=True,
                 use_val_ds=True,
                 num_retraining_epochs=10,
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

        # Look-Ahead
        self.use_true_labels = use_true_labels
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

    def select_batches(self, acq_size, unlabeled_features, labeled_features, labeled_labels, batch_types):
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
            elif batch_type == 'diverse':
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=acq_size, n_init='auto')
                clusters = km.fit_predict(unlabeled_features)
                indices = []
                for _ in range(num_batches):
                    idx = []
                    for i in range(acq_size):
                        indices_cluster = (clusters == i).nonzero()[0]
                        idx.append(self.rng.choice(indices_cluster))
                    indices.append(np.array(idx))
                indices_batches.extend(indices)
            elif batch_type == 'uncertain':
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression()
                clf.fit(labeled_features, labeled_labels.view(-1, 1))
                probas = clf.predict_proba(unlabeled_features)
                probas = torch.from_numpy(probas)
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
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])
        unlabeled_logits = model.get_logits_from_representations(unlabeled_features, device=self.device)

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_features = model.get_representations(labeled_dataloader, device=self.device)
        # labeled_logits = model.get_logits_from_representations(labeled_features, device=device).cpu()
        labeled_labels = torch.cat([batch[1] for batch in labeled_dataloader])

        indices_batches = self.select_batches(
            acq_size,
            unlabeled_features,
            labeled_features,
            labeled_labels,
            self.batch_types
        )

        loss_batches = []
        init_params = model.state_dict()
        for indices in track(indices_batches):
            features_batch = unlabeled_features[indices]

            if self.use_true_labels:
                # Use true labels of batch for model training
                labels = unlabeled_labels[indices]
                labels = labels.unsqueeze(0)
            else:
                # MC labels
                categorical = torch.distributions.Categorical(logits=unlabeled_logits[indices])
                labels = categorical.sample((self.num_mc_labels,))

            loss_labels = []
            for labels_batch in labels:
                if self.use_retraining:
                    model.reset_states(reset_model_parameters=True)
                    trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)

                    train_ds1 = Subset(al_datamodule.train_dataset, indices=labeled_indices)
                    train_ds2 = Subset_(
                        al_datamodule.train_dataset,
                        indices=np.array(unlabeled_indices)[indices],
                        labels=labels_batch
                    )
                    train_ds = ConcatDataset([train_ds1, train_ds2])
                    drop_last = len(train_ds) > al_datamodule.train_batch_size
                    train_loader = DataLoader(train_ds, al_datamodule.train_batch_size,
                                              shuffle=True, drop_last=drop_last)

                    trainer.fit(model, train_loader)
                else:
                    model.load_state_dict(init_params)
                    model.update_posterior(
                        iter(zip(features_batch, labels_batch)),
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


class Subset_(Subset):
    def __init__(self, dataset, indices, labels):
        super().__init__(dataset, indices)
        self.labels = labels
        assert len(indices) == len(labels)

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0], self.labels[idx]


if __name__ == '__main__':
    main()
