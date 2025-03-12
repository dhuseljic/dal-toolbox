import itertools

import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from lightning import Trainer
from collections import defaultdict
from rich.progress import track
from sklearn.linear_model import LogisticRegression
from dal_toolbox.active_learning.strategies.query import Query
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning.data import ActiveLearningDataModule


class PerfDALOracle(Query):
    def __init__(self,
                 al_strategies=['random', 'typiclust', 'dropquery', 'bait', 'typiclass', 'dropqueryclass', 'loss', 'margin', 'badge', 'coreset', 'alfamix'],
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
                 max_subset_size=5000,
                 vary_strat_subset_size=False,
                 device='cpu',
                 random_seed=None,
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device

        # Batch Selection
        self.strat_subset_size = strat_subset_size
        self.max_subset_size = max_subset_size
        self.vary_strat_subset_size = vary_strat_subset_size
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
        elif loss == 'brier':
            self.loss_fn = lambda logits, y: torch.mean(torch.sum((logits.softmax(-1) - torch.nn.functional.one_hot(y, num_classes=logits.shape[-1])) ** 2, dim=-1))
        else:
            raise NotImplementedError()

        # Retraining
        self.retraining = retraining
        self.num_retraining_epochs = num_retraining_epochs
        self.update_gamma = update_gamma

        # Noise filter
        self.i_iter = 0
        self.denoise_warmup = 5
        self.denoise_quantiles = np.linspace(.3, 1, self.denoise_warmup)

        # Some helper
        self.history = []

    def build_al_strategies(self, al_strategies):
        strategies_list = []
        for strat_name in al_strategies:
            if strat_name == 'random':
                strat = strategies.RandomSampling()
            elif strat_name == 'margin':
                strat = strategies.MarginSampling(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'coreset':
                strat = strategies.CoreSet(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'badge':
                # strat = GTBadge(subset_size=self.strat_subset_size, device=self.device)
                strat = strategies.Badge(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'typiclust':
                strat = strategies.TypiClust(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'alfamix':
                strat = strategies.AlfaMix(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'dropquery':
                strat = strategies.DropQuery(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'bait':
                strat = strategies.BaitSampling(subset_size=self.strat_subset_size,
                                                grad_likelihood='binary_cross_entropy', device=self.device)
            elif strat_name == 'typiclass':
                strat = TypiClass(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'dropqueryclass':
                strat = DropQueryClass(subset_size=self.strat_subset_size, device=self.device)
            elif strat_name == 'loss':
                strat = LossSampling(subset_size=self.strat_subset_size, device=self.device)
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

        base_loss = self.evaluate_model(model, al_datamodule.test_dataloader())

        indices_batches, batches_counts = self.select_strategy_batches(model, al_datamodule, acq_size, unlabeled_indices)

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
                        from_features=True,
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

    def select_strategy_batches(self, model, al_datamodule, acq_size, unlabeled_indices):
        batches = np.random.choice(self.batch_types, p=self.strat_ratio, size=self.num_batches)
        batches_counts = {t: np.sum(t == batches).item() for t in self.batch_types}

        indices = []
        for strat_name, strat in zip(self.batch_types, self.strategies):
            num_batches = batches_counts[strat_name]

            ss_min = acq_size*4
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
        indices = np.array(indices)
        return indices, batches_counts

    def filter_noisy_samples(self, al_datamodule, model):
        if self.i_iter > (self.denoise_warmup-1):
            return al_datamodule
        self.denoise_quantile = self.denoise_quantiles[self.i_iter]
        self.i_iter += 1
        al_dm = deepcopy(al_datamodule)

        u_dl, u_indices = al_dm.unlabeled_dataloader()
        u_outputs = model.get_model_outputs(u_dl, ['features', 'labels'], self.device)

        l_dl, l_indices = al_dm.labeled_dataloader()
        l_outputs = model.get_model_outputs(l_dl, ['features', 'labels'], self.device)

        # Train Model
        clf = LogisticRegression(C=.0001)
        all_features = torch.cat((u_outputs['features'], l_outputs['features']))
        all_labels = torch.cat((u_outputs['labels'], l_outputs['labels']))
        clf.fit(all_features, all_labels)
        # (clf.predict(all_features) == all_labels).float().mean()

        # Filter uncertain samples
        unlabeled_probas = clf.predict_proba(u_outputs['features'])
        unlabeled_entropy = - np.sum(unlabeled_probas * np.log(unlabeled_probas), axis=-1)
        thresh = np.quantile(unlabeled_entropy, q=self.denoise_quantile)
        indices = np.where(unlabeled_entropy <= thresh)[0]

        # Update unlabeled indices
        al_dm.unlabeled_indices = [al_dm.unlabeled_indices[idx] for idx in indices]

        return al_dm

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

    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)

        unlabeled_outputs = model.get_model_outputs(
            unlabeled_dataloader, output_types=['features', 'labels'], device=self.device)
        unlabeled_features = unlabeled_outputs['features']
        unlabeled_labels = unlabeled_outputs['labels']

        labeled_dataloader, _ = al_datamodule.labeled_dataloader()
        labeled_outputs = model.get_model_outputs(
            labeled_dataloader, output_types=['labels'], device=self.device)
        labeled_labels = labeled_outputs['labels']

        indices = select_samples_per_class(
            candidate_features=unlabeled_features,
            candidate_labels=unlabeled_labels,
            unlabeled_labels=unlabeled_labels,
            labeled_labels=labeled_labels,
            acq_size=acq_size
        )

        global_indices = [unlabeled_indices[idx] for idx in indices]
        return global_indices


class DropQueryClass(strategies.DropQuery):
    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_loader, _ = al_datamodule.labeled_dataloader()
        num_unlabeled = len(unlabeled_indices)

        unlabeled_outputs = model.get_model_outputs(
            unlabeled_loader, output_types=['features', 'logits', 'labels'], device=self.device)
        unlabeled_features = unlabeled_outputs['features']
        unlabeled_logits = unlabeled_outputs['logits']
        unlabeled_labels = unlabeled_outputs['labels']

        labeled_outputs = model.get_model_outputs(labeled_loader, output_types=['labels'], device=self.device)
        labeled_labels = labeled_outputs['labels']

        y_star = unlabeled_logits.softmax(-1).argmax(-1)
        candidates = self._get_candidates(model, unlabeled_features, y_star, acq_size)

        from dal_toolbox.active_learning.strategies.utils import get_random_samples
        if len(candidates) < acq_size:
            delta = acq_size - len(candidates)
            random_samples = get_random_samples(candidates, delta, num_unlabeled)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)
        else:
            candidate_vectors = F.normalize(unlabeled_features[candidates]).numpy()
            candidate_labels = unlabeled_labels[candidates]
            selected = select_samples_per_class(
                candidate_features=candidate_vectors,
                candidate_labels=candidate_labels,
                unlabeled_labels=unlabeled_labels,
                labeled_labels=labeled_labels,
                acq_size=acq_size,
            )

        selected_candidates = candidates[selected].tolist()
        query_indices = [unlabeled_indices[i] for i in selected_candidates]
        return query_indices


class LossSampling(Query):
    def __init__(self, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device

    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        unlabeled_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_loader, _ = al_datamodule.labeled_dataloader()

        unlabeled_outputs = model.get_model_outputs(
            unlabeled_loader, output_types=['features', 'logits', 'labels'], device=self.device)
        unlabeled_features = unlabeled_outputs['features']
        unlabeled_logits = unlabeled_outputs['logits']
        unlabeled_labels = unlabeled_outputs['labels']

        labeled_outputs = model.get_model_outputs(labeled_loader, output_types=['labels'], device=self.device)
        labeled_labels = labeled_outputs['labels']

        losses = torch.nn.functional.cross_entropy(unlabeled_logits, unlabeled_labels, reduction='none')
        quartile = losses.quantile(0.75)
        candidates = torch.where(losses > quartile)[0]
        selected = select_samples_per_class(
            candidate_features=unlabeled_features[candidates],
            candidate_labels=unlabeled_labels[candidates],
            unlabeled_labels=unlabeled_labels,
            labeled_labels=labeled_labels,
            acq_size=acq_size,
        )
        indices = candidates[selected]
        # indices = self.rng.choice(indices, size=acq_size, replace=False)
        return [unlabeled_indices[idx] for idx in indices]


def select_samples_per_class(candidate_features, candidate_labels, unlabeled_labels, labeled_labels, acq_size):
    from sklearn.metrics.pairwise import euclidean_distances
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


class CrossDomainOracle(Query):
    """Implementation of the oracle used in [1].

    [1] T. Werner et al. A Cross-Domain Benchmark for Active Learning. NeurIPS Datasets and Benchmarks Track, 2024.
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


class SimulatedAnnealingOracle(Query):
    """Implementation of the oracle used in [1].

    [1] Zhou, Yilun, et al. Towards understanding the behaviors of optimal deep active learning algorithms. AISTATS. 2021.
    """

    def __init__(self,
                 num_acq,
                 acq_size,
                 sa_steps=25000,
                 greedy_steps=5000,
                 linear_annealing_factor=0.1,
                 random_seed=None,
                 device='cpu',
                 ):
        super().__init__(random_seed)
        self.num_acq = num_acq
        self.acq_size = acq_size
        self.sa_steps = sa_steps
        self.greedy_steps = greedy_steps
        self.linear_annealing_factor = linear_annealing_factor
        self.device = device

        self.num_retraining_epochs = 200
        self.search_done = False
        self.i_acq = 0

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size: int):
        self.annealing_search(model, al_datamodule, acq_size)
        indices = self.queried_batches[self.i_acq]
        self.i_acq += 1
        return indices

    def annealing_search(self, model, al_datamodule, acq_size):
        if self.search_done:
            return
        _, u_indices = al_datamodule.unlabeled_dataloader()
        shuffle_idx = self.rng.permutation(len(u_indices))

        current_order = np.array(u_indices)[shuffle_idx]
        current_quality = self.quality(model, al_datamodule, order=current_order)

        best_order, best_quality = current_order, current_quality

        for t in track(range(self.sa_steps), 'Simulated Annealing Search'):
            new_order = self.propose_new_order(order=current_order)
            new_quality = self.quality(model, al_datamodule, order=new_order)

            delta_quality = (new_quality - current_quality)
            u = self.rng.rand()
            if u < np.exp(self.linear_annealing_factor * t * delta_quality):
                current_order, current_quality = new_order, new_quality
                if best_quality < current_quality:
                    best_order, best_quality = current_order, current_quality

        # Greedy refinement
        for t in track(range(self.greedy_steps), 'Greedy Refinement'):
            new_order = self.propose_new_order(best_order)
            new_quality = self.quality(model, al_datamodule, order=new_order)
            if new_quality > best_quality:
                best_order, best_quality = new_order, new_quality
        
        self.search_done = True
        self.queried_batches = [best_order[i_acq*acq_size:(i_acq+1)*acq_size] for i_acq in range(self.num_acq)]

    def propose_new_order(self, order):
        order = order.copy()
        swap_between_batches = (self.rng.rand() > 0.5) and self.num_acq > 1
        if swap_between_batches:  # Swap data point between batch
            b1, b2 = self.rng.randint(0, self.num_acq, size=2)
            while b1 == b2:
                b2 = self.rng.randint(0, self.num_acq)
            i1, i2 = self.rng.randint(0, self.acq_size, size=2)
            idx1 = self.acq_size*b1 + i1
            idx2 = self.acq_size*b2 + i2

        else:  # Swap data point from outside
            b1 = self.rng.randint(0, self.num_acq)
            i1 = self.rng.randint(0, self.acq_size)
            idx1 = self.acq_size*b1 + i1
            idx2 = self.rng.randint(self.acq_size*self.num_acq, len(order))

        order[idx1], order[idx2] = order[idx2], order[idx1]
        return order

    def quality(self, model, al_datamodule, order):
        learning_curves = defaultdict(list)

        _, l_indices = al_datamodule.labeled_dataloader()
        for i_acq in range(self.num_acq):
            new_indices = order[:(i_acq+1)*self.acq_size]

            model.reset_states(reset_model_parameters=True)
            retrain_indices = np.append(l_indices, new_indices)
            retrain_loader = al_datamodule.custom_dataloader(retrain_indices, train=True)
            trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
            trainer.fit(model, retrain_loader)

            acc = self.evaluate_model(model, al_datamodule.test_dataloader())
            learning_curves['accuracy'].append(acc)
        quality = np.mean(learning_curves['accuracy'])
        return quality

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.eval()
        model.to(self.device)
        num_samples = 0
        running_corrects = 0
        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            logits = model(inputs)

            num_samples += len(inputs)
            running_corrects += (logits.argmax(-1) == targets).sum().item()
        acc = running_corrects / num_samples
        return acc
