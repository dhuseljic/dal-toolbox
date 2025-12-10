import itertools

import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from rich.progress import track
from lightning import Trainer
from sklearn.metrics.pairwise import euclidean_distances

from .. import strategies
from ..data import ActiveLearningDataModule
from ...models.utils.callbacks import MetricLogger
from ..strategies.utils import get_random_samples


class BoSS(strategies.Query):
    def __init__(self,
                 strategies=['random', 'margin', 'coreset', 'badge', 'bait', 'typiclust', 'alfamix',
                             'dropquery', 'max_herding', 'uherding', 'typiclass', 'dropqueryclass'],
                 num_batches=100,
                 look_ahead='true_labels',
                 perf_estimation='test_ds',
                 retraining='train',
                 loss='zero_one',
                 random_seed=None,
                 device='cpu',
                 # Additional Args when deviating from GT
                 strat_ratio='equal',
                 num_retraining_epochs=50,
                 num_mc_labels=5,
                 update_gamma=10,
                 subset_size=None,
                 strat_subset_size=2500,
                 vary_strat_subset_size=False,
                 max_subset_size=5000,
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device
        self.strategies = self.build_al_strategies(strategies, strat_subset_size=strat_subset_size)
        self.num_batches = num_batches
        self.look_ahead = look_ahead
        self.perf_estimation = perf_estimation
        self.retraining = retraining
        self.loss_fn = get_loss_fn(loss_name=loss)


        # Additional arguments
        self.strat_subset_size = strat_subset_size
        self.max_subset_size = max_subset_size
        self.vary_strat_subset_size = vary_strat_subset_size
        self.batch_types = [type(s).__name__.lower() for s in self.strategies]
        self.batch_type_count = {k: 0 for k in self.batch_types}
        self.strat_ratio = strat_ratio
        if self.strat_ratio == 'equal':
            self.strat_ratio = np.ones(len(self.batch_types)) / len(self.batch_types)
        if len(self.strat_ratio) != len(self.batch_type_count):
            raise ValueError('Batch type ratio should be the same length as batch type count.')
        self.num_mc_labels = num_mc_labels
        self.num_retraining_epochs = num_retraining_epochs
        self.update_gamma = update_gamma

        self.i_iter = 0
        self.history = []

    def build_al_strategies(self, al_strategies, strat_subset_size=None):
        strategies_list = []
        for strat_name in al_strategies:
            if strat_name == 'random':
                strat = strategies.RandomSampling()
            elif strat_name == 'margin':
                strat = strategies.MarginSampling(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'coreset':
                strat = strategies.CoreSet(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'badge':
                strat = strategies.Badge(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'typiclust':
                strat = strategies.TypiClust(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'alfamix':
                strat = strategies.AlfaMix(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'max_herding':
                strat = strategies.MaxHerding(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'uherding':
                strat = strategies.UncertaintyHerding(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'dropquery':
                strat = strategies.DropQuery(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'bait':
                strat = strategies.BaitSampling(subset_size=strat_subset_size,
                                                grad_likelihood='binary_cross_entropy', device=self.device)
            elif strat_name == 'typiclass':
                strat = TypiClass(subset_size=strat_subset_size, device=self.device)
            elif strat_name == 'dropqueryclass':
                strat = DropQueryClass(subset_size=strat_subset_size, device=self.device)
            else:
                raise NotImplementedError()
            strategies_list.append(strat)
        return strategies_list

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        al_datamodule.unlabeled_indices = unlabeled_indices
        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, output_types=[
                                                    'logits', 'features', 'labels'], device=self.device)
        unlabeled_logits = unlabeled_outputs['logits']
        unlabeled_labels = unlabeled_outputs['labels']
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_outputs = model.get_model_outputs(labeled_dataloader, output_types=[
                                                  'features', 'labels'], device=self.device)
        labeled_labels = labeled_outputs['labels']

        if self.look_ahead == 'gt_model':
            if not hasattr(self, 'gt_model'):
                self.gt_model = deepcopy(model)
                indices = labeled_indices + unlabeled_indices
                train_loader = al_datamodule.custom_dataloader(indices, train=True)
                model.reset_states(reset_model_parameters=True)
                cb = [MetricLogger()]
                Trainer(barebones=True, max_epochs=100, callbacks=cb).fit(self.gt_model, train_loader)
            gt_outputs = self.gt_model.get_model_outputs(
                unlabeled_dataloader, output_types=['logits'], device=self.device)

        base_loss = self.evaluate_model(model, al_datamodule.test_dataloader())
        indices_batches, batches_counts = self.select_strategy_batches(
            model, al_datamodule, acq_size, unlabeled_indices)

        loss_batches = []
        init_params = model.state_dict()
        for indices in track(indices_batches):
            if self.look_ahead == 'true_labels':  # Use true labels of batch for model training
                labels = unlabeled_labels[indices]
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'gt_model':  # Use true labels of batch for model training
                gt_logits = gt_outputs['logits']
                labels = gt_logits[indices].argmax(-1)
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
                        from_features=True,
                        device=self.device
                    )
                else:
                    raise NotImplementedError()

                if self.perf_estimation == 'val_ds':
                    loss = self.evaluate_model(model, al_datamodule.val_dataloader())
                elif self.perf_estimation == 'gt_model':
                    out = model.get_model_outputs(unlabeled_dataloader, output_types=[
                                                  'logits'], device=self.device)
                    gt_model_labels = gt_outputs['logits'].argmax(-1)
                    loss = self.loss_fn(out['logits'], gt_model_labels)
                elif self.perf_estimation == 'test_ds':
                    loss = self.evaluate_model(model, al_datamodule.test_dataloader())
                elif self.perf_estimation == 'unlabeled_ds':
                    loss = self.evaluate_model(model, al_datamodule.unlabeled_dataloader()[0])
                elif self.perf_estimation == 'labeled_ds':
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

    def select_strategy_batches(self, model, al_datamodule, acq_size, unlabeled_indices):
        batches = np.random.choice(self.batch_types, p=self.strat_ratio, size=self.num_batches)
        batches_counts = {t: np.sum(t == batches).item() for t in self.batch_types}
        # TODO: This is for additional ablations, may remove later.  if self.one_batch_per_strat:

        indices = []
        for strat_name, strat in track(list(zip(self.batch_types, self.strategies)), "Selecting batches.."):
            num_batches = batches_counts[strat_name]

            # TODO: This is just a temporary solution for DTD as the unlabeled pool is very small
            ss_min = min(acq_size*4, len(al_datamodule.unlabeled_indices) - (num_batches+1))
            ss_max = min(len(al_datamodule.unlabeled_indices), self.max_subset_size)
            subset_range = self.rng.choice(range(ss_min, ss_max), size=num_batches, replace=False)

            indices_strat = []
            for i_batch in range(num_batches):
                if self.vary_strat_subset_size:
                    strat.subset_size = subset_range[i_batch]
                idx = strat.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
                indices_strat.append(idx)

            indices.extend(indices_strat)
        indices = np.array(indices)

        # Convert global indices from strategies to local indices
        masks = [np.isin(unlabeled_indices, idx) for idx in indices]
        indices = [np.where(mask)[0] for mask in masks]
        [len(idx) for idx in indices][-2]
        indices = np.array(indices)
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


class DropQueryClass(strategies.DropQuery):
    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        u_loader, u_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        l_loader, _ = al_datamodule.labeled_dataloader()
        num_unlabeled = len(u_indices)

        u_outputs = model.get_model_outputs(u_loader, ['features', 'logits', 'labels'], device=self.device)
        u_features = u_outputs['features']
        u_logits = u_outputs['logits']
        u_labels = u_outputs['labels']

        l_outputs = model.get_model_outputs(l_loader, output_types=['labels'], device=self.device)
        l_labels = l_outputs['labels']

        y_star = u_logits.softmax(-1).argmax(-1)
        candidates = self._get_candidates(model, u_features, y_star, acq_size)

        if len(candidates) < acq_size:
            delta = acq_size - len(candidates)
            random_samples = get_random_samples(candidates, delta, num_unlabeled)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)
        else:
            candidate_vectors = F.normalize(u_features[candidates]).numpy()
            candidate_labels = u_labels[candidates]
            selected = select_samples_per_class(
                candidate_features=candidate_vectors,
                candidate_labels=candidate_labels,
                unlabeled_labels=u_labels,
                labeled_labels=l_labels,
                acq_size=acq_size,
            )

        selected_candidates = candidates[selected].tolist()
        query_indices = [u_indices[i] for i in selected_candidates]
        return query_indices


class TypiClass(strategies.Query):
    def __init__(self, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device

    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        u_loader, u_indices = al_datamodule.unlabeled_dataloader(self.subset_size)

        u_outputs = model.get_model_outputs(u_loader, ['features', 'labels'], device=self.device)
        u_features = u_outputs['features']
        u_labels = u_outputs['labels']

        l_loader, _ = al_datamodule.labeled_dataloader()
        l_outputs = model.get_model_outputs(l_loader, ['labels'], device=self.device)
        l_labels = l_outputs['labels']

        indices = select_samples_per_class(
            candidate_features=u_features,
            candidate_labels=u_labels,
            unlabeled_labels=u_labels,
            labeled_labels=l_labels,
            acq_size=acq_size
        )
        global_indices = [u_indices[idx] for idx in indices]
        return global_indices


def select_samples_per_class(candidate_features, candidate_labels, unlabeled_labels, labeled_labels, acq_size):
    class_counter = {}
    all_labels = torch.cat((unlabeled_labels, labeled_labels))
    labels = all_labels.unique()
    for lbl in labels:
        class_counter[lbl.item()] = 0
    for lbl, count in zip(*labeled_labels.unique(return_counts=True)):
        class_counter[lbl.item()] = count.item()

    selected = []
    while len(selected) < acq_size:
        min_counts = min(class_counter.values())
        min_labels = [lbl for lbl, cnt in class_counter.items() if cnt == min_counts]
        for lbl in min_labels:
            if len(selected) == acq_size:
                break
            indices = torch.nonzero(candidate_labels == lbl).ravel()
            indices = indices[~np.isin(indices, selected)]
            if len(indices) < 1:
                class_counter.pop(lbl)
                continue
            dist_mat = euclidean_distances(candidate_features[indices.tolist()], squared=True)
            dist_sum = np.sum(dist_mat, axis=1)
            idx = dist_sum.argmin()

            selected.append(indices[idx].item())
            class_counter[lbl] += 1
    return selected


def get_loss_fn(loss_name):
    if loss_name == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_name == 'expected_cross_entropy':
        def loss_fn(logits, _): return torch.mean(- torch.sum(
            logits.softmax(-1) * logits.log_softmax(-1), dim=-1))
    elif loss_name == 'zero_one':
        def loss_fn(logits, y): return 1 - torch.mean((logits.argmax(dim=-1) == y).float())
    elif loss_name == 'expected_zero_one':
        def loss_fn(logits, _): return torch.mean(1 - logits.softmax(-1).max(-1).values)
    elif loss_name == 'brier':
        def loss_fn(logits, y): return torch.mean(
            torch.sum((logits.softmax(-1) - torch.nn.functional.one_hot(y, num_classes=logits.shape[-1])) ** 2, dim=-1))
    else:
        raise NotImplementedError()
    return loss_fn
