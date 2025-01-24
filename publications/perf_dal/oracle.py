import itertools

import torch
import numpy as np

from lightning import Trainer
from rich.progress import track
from dal_toolbox.active_learning.strategies.query import Query
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning.data import ActiveLearningDataModule


class PerfDALOracle(Query):
    def __init__(self,
                 al_strategies=['random', 'margin', 'typiclust', 'typiclass', 'bait', 'dropquery', 'alfamix'],
                 num_batches=200,
                 strat_ratio='equal',
                 look_ahead='true_labels',
                 num_mc_labels=5,
                 perf_estimation='val_ds',
                 retraining='train',
                 num_retraining_epochs=10,
                 update_gamma=10,
                 loss='cross_entropy',
                 subset_size=None,
                 strat_subset_size=2500,
                 device='cpu',
                 random_seed=None,
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device

        # Batch Selection
        self.strat_subset_size = strat_subset_size
        self.strategies = self.build_al_strategies(al_strategies)
        self.num_batches = num_batches
        self.batch_types = [type(s).__name__.lower() for s in self.strategies]
        self.batch_type_count = {k: 0 for k in self.batch_types}
        self.strat_ratio = strat_ratio
        if self.strat_ratio == 'equal':
            self.strat_ratio = np.ones(len(self.batch_types)) / len(self.batch_types)
        if len(self.strat_ratio) != len(self.batch_type_count):
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

    def build_al_strategies(self, al_strategies):
        strategies_list = []
        for strat_name in al_strategies:
            if strat_name == 'random':
                strat = strategies.RandomSampling()
            elif strat_name == 'margin':
                strat = strategies.MarginSampling(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'typiclust':
                strat = strategies.TypiClust(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'typiclass':
                strat = TypiClass(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'dropquery':
                strat = strategies.DropQuery(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'bait':
                strat = strategies.BaitSampling(subset_size=self.strat_subset_size,
                                                grad_likelihood='binary_cross_entropy', device=self.device)
            # elif strat == 'alfamix':
            #     strategies_list.append(strategies.AlfaMix(embed_dim=384,
            #         subset_size=self.strat_subset_size, device=self.device))
            else:
                raise NotImplementedError()
            strategies_list.append(strat)
        return strategies_list

    @ torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])
        unlabeled_logits = model.get_logits_from_representations(unlabeled_features, device=self.device)

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_features = model.get_representations(labeled_dataloader, device=self.device)
        labeled_labels = torch.cat([batch[1] for batch in labeled_dataloader])

        base_loss = self.evaluate_model(model, al_datamodule.test_dataloader())

        indices_batches, batches_counts = self.select_strategy_batches(model, al_datamodule, acq_size)

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

        # import pylab as plt
        # plt.figure()
        # for i, strat_name in enumerate(self.batch_types):
        #     start_idx = 0 if i == 0 else counts[i - 1]
        #     end_idx = counts[i]
        #     plt.hist(loss_batches[start_idx:end_idx], bins='auto', label=strat_name, alpha=0.5)
        # plt.vlines(base_loss, *plt.ylim(), lw=3, colors='k', ls='--', label='Base Loss')
        # plt.legend()
        # plt.savefig('tmp.png')

        self.history.append({
            'base_loss': base_loss,
            'loss_batches': np.array(loss_batches).tolist(),
            'bought_batch_type': batch_type,
            'batches_counts': batches_counts,
        })

        local_indices = indices_batches[best_idx]
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        return global_indices

    def select_strategy_batches(self, model, al_datamodule, acq_size):
        batches = np.random.choice(self.batch_types, p=self.strat_ratio, size=self.num_batches)
        batches_counts = {t: np.sum(t == batches).item() for t in self.batch_types}

        indices = []
        for strat_name, strat in zip(self.batch_types, self.strategies):
            num_batches = batches_counts[strat_name]
            indices_strat = [strat.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
                             for _ in range(num_batches)]
            indices.extend(indices_strat)
        indices = np.array(indices)

        # Convert global indices from strategies to local indices
        indices = np.array([np.where(np.isin(al_datamodule.unlabeled_indices, idx))[0] for idx in indices])

        return indices, batches_counts

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


class TypiClass(Query):
    def __init__(self, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device
        from collections import defaultdict
        self.class_counter = defaultdict(int)

    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        unlabeled_labels = torch.cat([batch[1] for batch in unlabeled_dataloader])

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_labels = torch.cat([batch[1] for batch in labeled_dataloader])

        all_labels = torch.cat((unlabeled_labels, labeled_labels))
        labels = all_labels.unique()
        for lbl in labels:
            self.class_counter[lbl.item()] = 0

        for lbl, count in zip(*labeled_labels.unique(return_counts=True)):
            self.class_counter[lbl.item()] = count.item()
        sorted_class_counter = dict(sorted(self.class_counter.items(), key=lambda x: x[1]))

        from itertools import cycle
        from dal_toolbox.active_learning.strategies.typiclust import calculate_typicality
        indices = []
        cycled_labels = cycle(sorted_class_counter.keys())
        while len(indices) != acq_size:
            lbl = next(cycled_labels)
            indices_lbl = np.where(unlabeled_labels == lbl)[0]
            candidates = indices_lbl[~np.isin(indices_lbl, indices)]
            if len(candidates) < 0:
                continue
            unlabeled_features_lbl = unlabeled_features[candidates]
            typicality = calculate_typicality(unlabeled_features_lbl, len(candidates)-1)
            idx = candidates[typicality.argmax()]
            # idx = self.rng.choice(candidates)
            indices.append(idx)
        indices = np.array(indices)

        global_indices = [unlabeled_indices[idx] for idx in indices]
        return global_indices

# TypiClass
# Sample high-loss samples
# Missclassifiations
# Gradient-Based
# Label Coverage


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

        self.margin = strategies.MarginSampling(device=self.device)
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
