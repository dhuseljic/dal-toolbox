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
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from rich.progress import track

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
    device = args.al.device
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'optimal':
        al_strategy = Optimal(
            subset_size=args.al.subset_size,
            num_batches=args.al.optimal.num_batches,
            num_mc_labels=args.al.optimal.num_mc_labels,
            gamma=args.al.optimal.gamma,
            loss=args.al.optimal.loss,
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
    def __init__(self, subset_size=None, num_batches=1000, num_mc_labels=1, loss='cross_entropy', gamma=1, device='cpu',  random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.num_batches = num_batches
        self.num_mc_labels = num_mc_labels
        self.gamma = gamma
        self.device = device

        if loss == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss == 'accuracy':
            self.loss_fn = lambda logits, y: 1 - torch.mean((logits.argmax(dim=-1) == y).float())
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):

        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)

        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])
        # unlabeled_logits = model.get_logits_from_representations(unlabeled_features, device=self.device)
        # labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        # labeled_features = model.get_representations(labeled_dataloader, device=device)
        # labeled_logits = model.get_logits_from_representations(labeled_features, device=device).cpu()

        # Select batches for argmin (random vs. diverse batches)
        indices_batches = [np.random.permutation(self.subset_size)[:acq_size]
                           for _ in range(self.num_batches)]

        optimal = True

        all_losses = []
        init_params = model.state_dict()
        for indices in track(indices_batches):
            features_batch = unlabeled_features[indices]

            # Expectation of labels 
            # Optimal: We just know the labels
            # Non-optimal: We need to evaluate the expectation
            if optimal:
                # Use true labels of batch for model training
                labels_batch = unlabeled_labels[indices]
            else:
                # Compute Expectation
                raise NotImplementedError()

            # Updating the model
            # Optimal: We just retrain with the new labels
            # Non-optimal: We use Bayesian updates
            if optimal:
                model.reset_states(reset_model_parameters=True)
                trainer = Trainer(barebones=True, max_epochs=50)
                aldm = copy.deepcopy(al_datamodule)
                aldm.update_annotations([unlabeled_indices[idx] for idx in indices])
                trainer.fit(model, aldm)
            else:
                model.load_state_dict(init_params)
                model.update_posterior(
                    iter(zip(features_batch, labels_batch)),
                    gamma=self.gamma,
                    from_representations=True,
                    device=self.device
                )

            # Evaluate the performance
            # Optimal: We just use a validation dataset
            # Non-Optimal: We need to approximate the performance without a validation dataset
            if optimal:
                model.to(self.device)
                num_samples = 0
                running_loss = 0
                test_loader = al_datamodule.test_dataloader()
                for batch in test_loader:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    logits = model(inputs)
                    num_samples += len(inputs)
                    running_loss += len(inputs)*self.loss_fn(logits, targets).item()
                loss = running_loss / num_samples
                all_losses.append(loss)
            else:
                raise NotImplementedError()

            # # MC sampling for one-step-lookahead
            # categorical = torch.distributions.Categorical(logits=unlabeled_logits[indices])
            # mc_labels = categorical.sample((self.num_mc_labels,))

            # mc_losses = []
            # for labels in mc_labels:
            #     # Incremental learning - p(y | x, L^+)
            #     features_batch = unlabeled_features[indices]
            #     model.load_state_dict(init_params)
            #     model.update_posterior(
            #         iter(zip(features_batch, labels)),
            #         gamma=self.gamma,
            #         from_representations=True,
            #         device=self.device
            #     )

            #     # TODO: check if predictions are changing before and after

            #     # Estimate new loss via validation set
            #     model.to(self.device)
            #     num_samples = 0
            #     running_loss = 0
            #     test_loader = al_datamodule.test_dataloader()
            #     for batch in test_loader:
            #         inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            #         logits = model(inputs)
            #         num_samples += len(inputs)
            #         running_loss += len(inputs)*self.loss_fn(logits, targets).item()

            #     mc_losses.append(running_loss / num_samples)
            # all_losses.append(np.mean(mc_losses))

        best_idx = np.argmin(all_losses)
        local_indices = indices_batches[best_idx]
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        return global_indices

        # combination_indices = np.array([np.random.permutation(self.subset_size)[:acq_size] for i in range(self.num_combinations)])
        # # Compute distances from
        # distances = []
        # for batch_idx in self.batch_indices:
        #     mask_batch = np.isin(np.array(labeled_indices), np.array(batch_idx))
        #     X_batch = labeled_features[mask_batch]
        #     dist = [batch_distance(X_batch, unlabeled_features[comb_idx]) for comb_idx in combination_indices]
        #     distances.append(dist)
        # np.array(distances).shape
        # idx = np.argmax(np.min(distances, axis=0))
        # local_indices = combination_indices[idx]


def batch_distance(X1, X2):
    cost_matrix = pairwise_distances(X1, X2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.sum(cost_matrix[row_ind, col_ind])


if __name__ == '__main__':
    main()
