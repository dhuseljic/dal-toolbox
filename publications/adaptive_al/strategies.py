import copy
import torch
import numpy as np
from lightning import Trainer
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from rich.progress import track
from sklearn.model_selection import KFold


class SelectAL(Query):
    def __init__(self,
                 epsilon=.5,
                 low_budget_strategy='typiclust',
                 high_budget_strategy='badge',
                 surrogate_low_strategy='typiclust',
                 surrogate_high_strategy='badge',
                 num_val_reps=10,
                 train_epochs=200,
                 subset_size=None,
                 random_seed=None,
                 device='cpu',
                 ):
        super().__init__(random_seed)
        self.epsilon = epsilon
        self.subset_size = subset_size
        self.device = device
        self.num_val_reps = num_val_reps
        self.train_epochs = train_epochs
        self.history = []

        self.random_strategy = strategies.RandomSampling()
        self.surrogate_low_strategy = build_al_strategies([surrogate_low_strategy], device=self.device)[0]
        self.surrogate_high_strategy = build_al_strategies([surrogate_high_strategy], device=self.device)[0]
        self.surrogate_strategies = [self.random_strategy,
                                     self.surrogate_low_strategy, self.surrogate_high_strategy]

        self.low_budget_strategy = build_al_strategies([low_budget_strategy], device=self.device)[0]
        self.high_budget_strategy = build_al_strategies([high_budget_strategy], device=self.device)[0]
        self.strategies = [self.random_strategy, self.low_budget_strategy, self.high_budget_strategy]

    def query(self, *, model, al_datamodule, acq_size):
        al_datamodule = copy.deepcopy(al_datamodule)
        _, labeled_indices = al_datamodule.labeled_dataloader()
        if self.epsilon >= len(labeled_indices):
            raise ValueError(
                f'Epsilon={self.epsilon} greater or equals the labeled pool size of {len(labeled_indices)}.')

        # 1. Determine the regime we are in
        labeled_labels = torch.cat([batch[1] for batch in al_datamodule.custom_dataloader(labeled_indices)])
        labels_unique, labels_counts = labeled_labels.unique(return_counts=True)
        num_classes = len(labels_unique)

        min_lbl_count = labels_counts.min()
        eps = self.epsilon if not (0 < self.epsilon < 1) else int(self.epsilon*len(labeled_indices))
        c = max(eps // num_classes, 1)
        if c > min_lbl_count:
            c = min_lbl_count

        surrogate_accs = []
        for strat in self.surrogate_strategies:
            # Select from L via strat per class
            remove_indices = []
            for lbl in labels_unique:
                labeled_indices_cls = np.array(labeled_indices)[labeled_labels == lbl]
                aldm = copy.deepcopy(al_datamodule)
                aldm.unlabeled_indices = labeled_indices_cls
                aldm.labeled_indices = []
                remove_indices.extend(strat.query(model=model, al_datamodule=aldm, acq_size=c))

            # Remove selection from L
            new_indices = copy.copy(labeled_indices)
            for idx in remove_indices:
                new_indices.remove(idx)

            # Eval the performance via cross validation on new labeled data
            accs = []
            num_val_samples = max(1, int(len(new_indices)*0.01))
            for _ in range(self.num_val_reps):
                val_indices = self.rng.choice(new_indices, size=num_val_samples, replace=False)
                train_indices = np.setdiff1d(new_indices, val_indices)
                model = self.train_model(model, aldm.custom_dataloader(train_indices, train=True))
                acc = self.evaluate_model(model, aldm.custom_dataloader(val_indices)).item()
                accs.append(acc)
            surrogate_accs.append(np.mean(accs).item())
        idx = self.rng.choice(np.flatnonzero(surrogate_accs == np.min(surrogate_accs)))

        strat = self.strategies[idx]
        u_indices = self.rng.choice(al_datamodule.unlabeled_indices, size=self.subset_size, replace=False)
        al_datamodule.unlabeled_indices = u_indices
        query_indices = strat.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
        return query_indices

    @torch.enable_grad()
    def train_model(self, model, dataloader):
        model.reset_states(reset_model_parameters=True)
        trainer = Trainer(barebones=True, max_epochs=self.train_epochs)
        trainer.fit(model, dataloader)
        return model

    @torch.no_grad()
    def evaluate_model(self, model, dataloader):
        model.eval()
        model.to(self.device)
        num_samples = 0
        running_corrects = 0
        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            pred = model(inputs).argmax(-1)

            num_samples += len(inputs)
            running_corrects += (pred == targets).float().sum()
        acc = running_corrects / num_samples
        return acc


class AdaptiveAL(Query):
    def __init__(self,
                 al_strategies=['random', 'margin', 'badge', 'bait', 'typiclust', 'alfamix', 'dropquery'],
                 num_batches=100,
                 strat_ratio='equal',
                 perf_estimation='val_ds',
                 num_mc_labels=5,
                 num_retraining_epochs=50,
                 look_ahead='true_labels',
                 subset_size=None,
                 max_subset_size=5000,
                 device='cpu',
                 random_seed=None,
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device

        self.num_batches = num_batches
        self.subset_size = subset_size
        self.max_subset_size = max_subset_size
        self.strat_ratio = strat_ratio
        self.look_ahead = look_ahead
        self.num_mc_labels = num_mc_labels
        self.num_retraining_epochs = num_retraining_epochs
        self.perf_erstimation = perf_estimation
        self.strategies = build_al_strategies(al_strategies)

        self.batch_types = [type(s).__name__.lower() for s in self.strategies]
        self.batch_type_count = {k: 0 for k in self.batch_types}
        if self.strat_ratio == 'equal':
            self.strat_ratio = np.ones(len(self.batch_types)) / len(self.batch_types)
        if len(self.strat_ratio) != len(self.batch_type_count):
            raise ValueError('Batch type ratio should be the same length as batch type count.')

        self.loss_fn = lambda logits, y: 1 - torch.mean((logits.argmax(dim=-1) == y).float())
        self.val_loader = None

        self.i_iter = 0
        self.history = []

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        al_datamodule = copy.deepcopy(al_datamodule)
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader()
        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, output_types=[
                                                    'logits', 'features', 'labels'], device=self.device)
        unlabeled_logits = unlabeled_outputs['logits']
        unlabeled_labels = unlabeled_outputs['labels']

        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()
        labeled_outputs = model.get_model_outputs(labeled_dataloader, output_types=[
                                                  'features', 'labels'], device=self.device)
        labeled_labels = labeled_outputs['labels']

        if self.val_loader is None:
            # Sample representative validation dataset
            num_val_samples = 50
            typiclust = strategies.TypiClust(device=self.device)
            val_indices = typiclust.query(model=model, al_datamodule=al_datamodule, acq_size=num_val_samples)
            self.val_loader = al_datamodule.custom_dataloader(val_indices)

        # Evaluate current performance
        base_loss = self.evaluate_model(model, self.val_loader)
        indices_batches, batches_counts = self.select_strategy_batches(
            model, al_datamodule, acq_size, unlabeled_indices)

        loss_batches = []
        for indices in track(indices_batches, "Evaluating influence of candidate batches.."):
            if self.look_ahead == 'true_labels':  # Use true labels of batch for model training
                labels = unlabeled_labels[indices]
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'pseudo_labels':  # Use pseudo labels of model
                logits = unlabeled_logits[indices]
                labels = logits.argmax(-1)
                labels = labels.unsqueeze(0)
            else:
                raise NotImplementedError()

            loss_labels = []
            for labels_batch in labels:
                if isinstance(labels_batch, tuple):
                    labels_batch = torch.Tensor(labels_batch).long()

                model.reset_states(reset_model_parameters=True)
                retrain_indices = labeled_indices + [unlabeled_indices[idx] for idx in indices]
                custom_labels = torch.cat((labeled_labels, labels_batch))
                retrain_loader = al_datamodule.custom_dataloader(
                    indices=retrain_indices, train=True, custom_labels=custom_labels)
                trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
                trainer.fit(model, retrain_loader)

                loss = self.evaluate_model(model, self.val_loader)

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
        for strat_name, strat in track(list(zip(self.batch_types, self.strategies)), "Sampling candidate batches.."):
            num_batches = batches_counts[strat_name]

            indices_strat = []
            for i_batch in range(num_batches):
                aldm = copy.deepcopy(al_datamodule)
                _, u_indices = aldm.unlabeled_dataloader(self.subset_size)
                aldm.unlabeled_indices = u_indices

                idx = strat.query(model=model, al_datamodule=aldm, acq_size=acq_size)
                indices_strat.append(idx)

            indices.extend(indices_strat)
        indices = np.array(indices)

        # Convert global indices from strategies to local indices
        masks = [np.isin(unlabeled_indices, idx) for idx in indices]
        indices = [np.where(mask)[0] for mask in masks]
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


def build_al_strategies(al_strategies, device='cpu'):
    strategies_list = []
    for strat_name in al_strategies:
        if strat_name == 'random':
            strat = strategies.RandomSampling()
        elif strat_name == 'margin':
            strat = strategies.MarginSampling(device=device)
        elif strat_name == 'coreset':
            strat = strategies.CoreSet(device=device)
        elif strat_name == 'badge':
            strat = strategies.Badge(device=device)
        elif strat_name == 'typiclust':
            strat = strategies.TypiClust(device=device)
        elif strat_name == 'alfamix':
            strat = strategies.AlfaMix(device=device)
        elif strat_name == 'dropquery':
            strat = strategies.DropQuery(device=device)
        elif strat_name == 'bait':
            strat = strategies.BaitSampling(grad_likelihood='binary_cross_entropy', device=device)
        else:
            raise NotImplementedError()
        strategies_list.append(strat)
    return strategies_list


class ActiveLearningByLearning(Query):
    def __init__(self, budget, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.budget = budget
        self.subset_size = subset_size
        self.device = device

        self.strategies = [
            strategies.RandomSampling(),
            strategies.MarginSampling(device=self.device),
        ]

        self.weights = np.ones(len(self.strategies))
        self.num_arms = self.num_experts = len(self.strategies)
        self.iterations = 100
        self.delta = 0.1
        self.p_min = np.sqrt(np.log(self.num_arms) / self.num_experts / self.iterations)

        self.query_history = []
        self.importance_weights = []

    def query(self, *, model, al_datamodule, acq_size):
        _, u_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        _, l_indices = al_datamodule.labeled_dataloader()
        all_indices = u_indices + l_indices

        if len(self.query_history) > 0:
            reward = self.compute_reward_(model, al_datamodule, all_indices)
            r_hat = reward * self.queried_instances[:, self.local_idx] / self.q[self.local_idx]
            v_hat = 1 / self.p
            self.weights = self.weights * np.exp(self.p_min / 2 * (
                r_hat + v_hat * np.sqrt(
                    np.log(self.num_arms / self.delta) / self.num_experts / self.budget
                )
            )
            )

        W = np.sum(self.weights)
        p = (1 - self.num_arms * self.p_min) * self.weights / W + self.p_min

        # Each strategy needs to select instances based on their heuristic
        psi = np.zeros((len(self.strategies), len(u_indices) + len(l_indices)))
        for i_strat, strat in enumerate(self.strategies):
            aldm = copy.deepcopy(al_datamodule)
            aldm.unlabeled_indices = all_indices

            global_idx = strat.query(model=model, al_datamodule=aldm, acq_size=acq_size)[0]
            local_idx = np.isin(all_indices, global_idx).nonzero()[0]
            psi[i_strat, local_idx] = 1
        q = np.dot(p, psi)

        local_idx = np.random.choice(range(len(all_indices)), size=1, p=q)[0]
        global_idx = all_indices[local_idx]
        global_q = q[local_idx]

        self.local_idx = local_idx
        self.p = p
        self.q = q
        self.queried_instances = psi
        self.query_history.append(global_idx)
        self.importance_weights.append(1 / global_q)

        if global_idx in l_indices:
            raise NotImplementedError()

        return [global_idx]

    @torch.no_grad()
    def compute_reward_(self, model, aldm, all_indices):
        model.eval()
        model.to(self.device)
        dataloader = aldm.custom_dataloader(indices=self.query_history)

        iw_corrects = 0
        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            preds = model(inputs).argmax(dim=-1)
            iw_corrects += np.sum(self.importance_weights * (preds == targets).numpy())
        iw_accuracy = iw_corrects / (len(all_indices)*self.budget)
        return iw_accuracy
