import itertools

import torch
import numpy as np

from lightning import Trainer
from rich.progress import track
from dal_toolbox.active_learning.strategies.query import Query
from dal_toolbox.active_learning.strategies import MarginSampling
from dal_toolbox.active_learning.data import ActiveLearningDataModule


class CrossDomainOracle(Query):
    """Implementation of the oracle used in [1].

    [1] T. Werner, J. Burchert, M. Stubbemann, and L. Schmidt-Thieme, “A Cross-Domain Benchmark for Active Learning,” in Neural Information Processing Systems Datasets and Benchmarks Track, 2024.
    """

    def __init__(self,
                 tau=20,
                 num_retraining_epochs=50,
                 eval_ds='test',
                 loss='cross_entropy',
                 random_seed=None,
                 device='cpu',
                 ):
        super().__init__(random_seed)
        self.tau = tau
        self.num_retraining_epochs = num_retraining_epochs
        self.eval_ds = eval_ds
        self.device = device

        self.margin = MarginSampling(device=self.device)
        if loss == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            NotImplementedError()

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size: int):
        if acq_size != 1:
            raise ValueError('The oracle is only defined for an acquisition size of 1.')
        if self.eval_ds == 'val':
            eval_dataloader = al_datamodule.val_dataloader()
        elif self.eval_ds == 'test':
            eval_dataloader = al_datamodule.test_dataloader()
        else:
            raise NotImplementedError(f'Evaluation dataset {self.eval_ds} not implemented')
        _, unlabeded_indices = al_datamodule.unlabeled_dataloader()
        _, labeled_indices = al_datamodule.labeled_dataloader()

        base_loss = self.evaluate_model(model, eval_dataloader)

        indices = self.rng.choice(unlabeded_indices, replace=False, size=self.tau)

        all_losses = []
        for idx in indices:
            retrain_indices = np.append(labeled_indices, idx)
            model.reset_states(reset_model_parameters=True)
            retrain_loader = al_datamodule.custom_dataloader(indices=retrain_indices, train=True)
            trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
            trainer.fit(model, retrain_loader)

            loss = self.evaluate_model(model, eval_dataloader)
            all_losses.append(loss)

        if np.any(np.less(all_losses, base_loss)):
            global_idx = indices[np.argmin(all_losses)]
            global_idx = [global_idx]
        else:
            global_idx = self.margin.query(model=model, al_datamodule=al_datamodule, acq_size=1)
        return global_idx

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.eval()
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


class PerfDALOracle(Query):
    def __init__(self,
                 num_batches=200,
                 batch_types=['random', 'diverse', 'uncertain'],
                 batch_types_ratio='equal',
                 look_ahead='true_labels',
                 num_mc_labels=5,
                 perf_estimation='val_ds',
                 retraining='train',
                 num_retraining_epochs=10,
                 update_gamma=10,
                 loss='cross_entropy',
                 subset_size=None,
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
        self.batch_types_ratio = batch_types_ratio
        if self.batch_types_ratio == 'equal':
            self.batch_types_ratio = np.ones(len(self.batch_types)) / len(self.batch_types)
        if len(self.batch_types_ratio) != len(self.batch_type_count):
            raise ValueError('Batch type ratio should be the same length as batch type count.')
        
        # Look-Ahead - True labels, All comb, MC, Pseudo Labels
        self.look_ahead = look_ahead
        self.num_mc_labels = num_mc_labels

        # Performance Estimation
        self.perf_erstimation = perf_estimation
        if loss == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss == 'expected_cross_entropy':
            self.loss_fn = lambda logits, _: torch.mean(- torch.sum(
                logits.softmax(-1) * logits.log_softmax(-1), dim=-1))
        elif loss == 'zero_one':
            self.loss_fn = lambda logits, y: 1 - torch.mean((logits.argmax(dim=-1) == y).float())
        elif loss == 'expected_zero_one':
            self.loss_fn = lambda logits, _: torch.mean(1 - logits.softmax(-1).max(-1).values)
        else:
            raise NotImplementedError()

        # Retraining
        self.retraining = retraining
        self.num_retraining_epochs = num_retraining_epochs
        self.update_gamma = update_gamma

        # Some helper
        self.history = []

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])
        unlabeled_logits = model.get_logits_from_representations(unlabeled_features, device=self.device)

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_features = model.get_representations(labeled_dataloader, device=self.device)
        labeled_labels = torch.cat([batch[1] for batch in labeled_dataloader])

        base_loss = self.evaluate_model(model, al_datamodule.test_dataloader())

        indices_batches, batches_counts = self.select_batches(
            acq_size,
            unlabeled_features,
            unlabeled_logits,
            unlabeled_labels,
            labeled_features,
            labeled_labels,
            self.batch_types
        )

        loss_batches = []
        init_params = model.state_dict()
        for indices in track(indices_batches):
            if self.look_ahead == 'true_labels':  # Use true labels of batch for model training
                labels = unlabeled_labels[indices]
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'pseudo_labels':  # Use pseudo labels of model
                logits = unlabeled_logits[indices]
                labels = logits.argmax(-1)
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'mc_labels':  # Samples labels via Monte Carlo
                categorical = torch.distributions.Categorical(logits=unlabeled_logits[indices])
                labels = categorical.sample((self.num_mc_labels,))
            elif self.look_ahead == 'all_labels':  # Go through all label combinations
                num_classes = unlabeled_logits.size(-1)
                labels = itertools.product(range(num_classes), repeat=acq_size)
            else:
                raise NotImplementedError()

            loss_labels = []
            for labels_batch in labels:
                if isinstance(labels_batch, tuple):
                    labels_batch = torch.Tensor(labels_batch).long()

                if self.retraining == 'train':
                    model.reset_states(reset_model_parameters=True)
                    retrain_indices = labeled_indices + [unlabeled_indices[idx] for idx in indices]
                    custom_labels = torch.cat((labeled_labels, labels_batch))
                    retrain_loader = al_datamodule.custom_dataloader(
                        indices=retrain_indices, train=True, custom_labels=custom_labels)
                    trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
                    trainer.fit(model, retrain_loader)

                elif self.retraining == 'update':
                    model.load_state_dict(init_params)
                    retrain_indices = [unlabeled_indices[idx] for idx in indices]
                    retrain_loader = al_datamodule.custom_dataloader(
                        indices=retrain_indices, train=True, custom_labels=labels_batch)
                    model.update_posterior(
                        retrain_loader,
                        gamma=self.update_gamma,
                        from_representations=True,
                        device=self.device
                    )
                else:
                    raise NotImplementedError()

                if self.perf_erstimation == 'val_ds':
                    loss = self.evaluate_model(model, al_datamodule.val_dataloader())
                elif self.perf_erstimation == 'test_ds':
                    loss = self.evaluate_model(model, al_datamodule.test_dataloader())
                elif self.perf_erstimation == 'unlabeled_ds':
                    loss = self.evaluate_model(model, al_datamodule.unlabeled_dataloader()[0])
                elif self.perf_erstimation == 'labeled_ds':
                    loss = self.evaluate_model(model, al_datamodule.labeled_dataloader()[0])
                else:
                    raise NotImplementedError(
                        f'Performance estimation {self.perf_estimation} not implemented.')

                loss_labels.append(loss)
            loss_batches.append(np.mean(loss_labels))
        best_idx = np.argmin(loss_batches)

        counts = np.cumsum(list(batches_counts.values()))
        idx_batch_type = np.searchsorted(counts, best_idx, side='right').item()
        batch_type = self.batch_types[idx_batch_type]
        self.batch_type_count[batch_type] += 1

        self.history.append({
            'base_loss': base_loss,
            'loss_batches': np.array(loss_batches).tolist(),
            'bought_batch_type': batch_type,
            'batches_counts': batches_counts,
        })

        local_indices = indices_batches[best_idx]
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        return global_indices

    def select_batches(self, acq_size, unlabeled_features, unlabeled_logits, unlabeled_labels, labeled_features, labeled_labels, batch_types):
        # TODO: maybe focus more often on clusters with many samples? class distribution
        batches = np.random.choice(self.batch_types, p=self.batch_types_ratio, size=self.num_batches)
        batches_counts = {t: np.sum(t == batches).item() for t in self.batch_types}

        indices_batches = []
        for batch_type in batch_types:
            num_batches = batches_counts[batch_type]
            if batch_type == 'random':
                indices = [np.random.permutation(len(unlabeled_features))[:acq_size]
                           for _ in range(num_batches)]
                indices_batches.extend(indices)

            elif batch_type == 'diverse':  # KMeans with covered and uncovered clusters
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
                probas = unlabeled_logits.softmax(-1)
                top_probas, _ = torch.topk(probas, k=2, dim=-1)
                margin_uncertainty = 1 - (top_probas[:, 0] - top_probas[:, 1])
                indices_difficult = np.where(margin_uncertainty > margin_uncertainty.quantile(.9))[0]
                indices = [self.rng.choice(indices_difficult, size=acq_size, replace=False)
                           for _ in range(num_batches)]
                indices_batches.extend(indices)
            else:
                raise ValueError(f'Batch type {batch_type} not implemented')

        return np.array(indices_batches), batches_counts

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.eval()
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
