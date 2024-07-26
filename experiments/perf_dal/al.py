import time
import hydra
import torch
import mlflow
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
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

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
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    if args.al.init_method == 'random':
        al_datamodule.random_init(n_samples=args.al.num_init_samples)
    elif args.al.init_method == 'diverse_dense':
        al_datamodule.diverse_dense_init(n_samples=args.al.num_init_samples)
    elif args.al.init_method == 'none':
        pass
    else:
        raise NotImplementedError()
    al_strategy = build_al_strategy(args)

    history = []
    num_features = 384
    model = build_model(args, num_features=num_features, num_classes=num_classes)
    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        barebones=True,
        callbacks=[MetricLogger()],
        # default_root_dir=args.output_dir,
        # enable_checkpointing=False,
        # logger=False,
        # enable_progress_bar=False,
    )
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
        trainer.fit(model, train_dataloaders=al_datamodule)

        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0
        print(f'Cycle {i_acq}:', test_stats, flush=True)
        history.append(test_stats)

    mlflow.set_tracking_uri(uri="{}".format(args.mlflow_uri))
    experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(history):
        mlflow.log_metrics(test_stats, step=i_acq)
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
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'diverse_batches':
        al_strategy = DiverseBatches(subset_size=args.al.subset_size)
    elif args.al.strategy == 'margin':
        al_strategy = strategies.MarginSampling(subset_size=args.al.subset_size)
    elif args.al.strategy == 'badge':
        al_strategy = strategies.Badge(subset_size=args.al.subset_size)
    else:
        raise NotImplementedError()
    return al_strategy


class DiverseBatches(Query):
    def __init__(self, subset_size=None, random_seed=None, num_combinations=100, strat='test'):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.init_run = True
        self.num_combinations = num_combinations
        self.strat = strat

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        if self.init_run:
            # Save the first indices
            self.batch_indices = [labeled_indices]
            self.init_run = False
        features_unlabeled, logits_unlabeled = model.get_representations_and_logits(unlabeled_dataloader)
        features_labeled, logits_labeled = model.get_representations_and_logits(labeled_dataloader)

        combination_indices = np.array([np.random.permutation(
            self.subset_size)[:acq_size] for i in range(self.num_combinations)])

        # Compute distances from
        distances = []
        for batch_idx in self.batch_indices:
            mask_batch = np.isin(np.array(labeled_indices), np.array(batch_idx))
            X_batch = features_labeled[mask_batch]
            dist = [batch_distance(X_batch, features_unlabeled[comb_idx]) for comb_idx in combination_indices]
            distances.append(dist)
        np.array(distances).shape

        if self.strat is None:
            idx = np.argmax(np.min(distances, axis=0))
            local_indices = combination_indices[idx]
        else:
            from dal_toolbox.active_learning.strategies.uncertainty import margin_score
            sort_idx = np.argsort(np.min(distances, axis=0))[::-1]
            combination_indices_sorted = combination_indices[sort_idx]

            comb_scores = []
            for comb_idx in combination_indices_sorted[:10]:
                logits_comb = logits_unlabeled[comb_idx]
                probas_comb = logits_comb.softmax(-1)
                scores = margin_score(probas_comb)
                score = scores.sum().item()
                comb_scores.append(score)
            best_comb_idx = np.argmax(comb_scores)
            local_indices = combination_indices_sorted[best_comb_idx]
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        self.batch_indices.append(global_indices)
        return global_indices


def batch_distance(X1, X2):
    cost_matrix = pairwise_distances(X1, X2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.sum(cost_matrix[row_ind, col_ind])


if __name__ == '__main__':
    main()
