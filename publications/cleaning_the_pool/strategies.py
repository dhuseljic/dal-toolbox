import copy
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from lightning import Trainer
from lightning.pytorch import Callback
from scipy.stats import spearmanr
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.active_learning.data import ActiveLearningDataModule
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from dal_toolbox.metrics.calibration import ExpectedCalibrationError
from dal_toolbox.active_learning.strategies.bait import get_exp_grad_representations
from scipy.stats import rankdata, qmc
from rich.progress import track
from torch.optim.swa_utils import AveragedModel
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
import math
import random
from torch.utils.data import TensorDataset, DataLoader, Subset
import warnings
from utils import build_model
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning


class Refine(Query):
    def __init__(self,
                 al_strategies=['random', 'margin', 'badge', 'bait', 'typiclust',
                                'alfamix', 'dropquery', 'max_herding', 'unc_herding'],
                 progressive_depth=5,
                 num_batches=100,
                 alpha=0.4,
                 select_strategy='unc_herding',
                 init_subset_size=5000,
                 max_pool_size=10000,
                 filter_acq_size=None,
                 # EER
                 perf_estimation='unlabeled_pool',
                 look_ahead='mc_labels',
                 loss='zero_one',
                 num_mc_labels=10,
                 ema_lmb=0.0,
                 eval_gt=False,
                 temp_scaling=False,
                 num_retraining_epochs=50,
                 device='cpu',
                 strat_ratio='equal',
                 label_noise_rate=0.0,
                 val_noise_rate=0.0,
                 random_seed=None,
                 ):
        super().__init__(random_seed=random_seed)
        self.device = device

        # Main Hyperparameters
        self.progressive_depth = progressive_depth
        self.num_batches = num_batches
        self.alpha = alpha
        self.init_subset_size = init_subset_size
        self.max_pool_size = max_pool_size
        self.filter_acq_size = filter_acq_size
        self.strategies = build_al_strategies(al_strategies)
        self.select_strategy = build_al_strategies([select_strategy])[select_strategy]

        self.strat_ratio = strat_ratio
        if self.strat_ratio == 'equal':
            self.strat_ratio = np.ones(len(self.strategies)) / len(self.strategies)

        # EER
        self.loss = loss
        self.eval_gt = eval_gt
        self.look_ahead = look_ahead
        self.num_mc_labels = num_mc_labels
        self.temp_scaling = temp_scaling
        self.ema_lmb = ema_lmb
        self.ema_scores = None
        self.perf_estimation = perf_estimation
        self.num_retraining_epochs = num_retraining_epochs
        self.label_noise_rate = label_noise_rate
        self.val_noise_rate = val_noise_rate
        self.compute_ece = ExpectedCalibrationError()

        self.strat_buy_counter = {k: 0 for k in self.strategies}
        if len(self.strat_ratio) != len(self.strat_buy_counter):
            raise ValueError('Batch type ratio should be the same length as batch type count.')

        self.iter = 0
        self.history = defaultdict(list)
        self.all_scores = defaultdict(list)
        self.all_rankings = defaultdict(list)
        self.all_gt_scores = defaultdict(list)
        self.all_gt_rankings = defaultdict(list)

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        self.iter += 1
        al_datamodule = copy.deepcopy(al_datamodule)
        model = copy.deepcopy(model)

        # def get_alpha(cycle_t, total_cycles, alpha_min=0.2, alpha_max=0.8):
        #     # Linear: return alpha_min + (alpha_max - alpha_min) * (cycle_t / total_cycles)
        #     return alpha_min + (alpha_max - alpha_min) * (1 - np.exp(-3 * cycle_t / total_cycles))
        # alpha = get_alpha(self.iter-1, 20)
        # self.history['alpha'].append(alpha.item())

        pools = {}
        pool_indices = {}
        new_pool_indices = None
        new_subset_size = self.init_subset_size
        for i in range(self.progressive_depth):
            indices_batches = self._preselect_batches(
                model,
                al_datamodule,
                acq_size,
                subset_size=new_subset_size,
                u_indices=new_pool_indices
            )
            new_pool_indices = np.unique(indices_batches)
            if len(new_pool_indices) > self.max_pool_size:
                new_pool_indices = self.rng.choice(new_pool_indices, self.max_pool_size, replace=False)
            new_subset_size = max(acq_size, int(self.alpha * len(new_pool_indices)))
            pool_indices[i] = new_pool_indices.tolist()
            pools[i+1] = new_pool_indices
        print("Pool Sizes:", {k: len(v) for k, v in pools.items()})
        for pool_idx in pools:
            key = f'pool_{pool_idx}'
            self.history[key].append(len(pools[pool_idx]))
        self.pool_indices = pool_indices

        if self.progressive_depth > 0:
            al_datamodule.unlabeled_indices = pools[self.progressive_depth].tolist()

        if self.select_strategy is not None:
            indices = self.select_strategy.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
            return indices

        indices_batches = [self.strategies[strat_name].query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
                           for strat_name in self.strategies]
        indices_batches = np.array(indices_batches)

        self.val_indices = pools[self.progressive_depth]
        if self.loss in ['fir_kfac', 'fir_binary']:
            scores_batches = self._estimate_influence_fisher(model, al_datamodule, indices_batches, eps=3)
        elif self.loss in ['zero_one', 'cross_entropy', 'moc', 'mocu', 'kl']:
            scores_batches = self._estimate_expected_error(model, al_datamodule, indices_batches)
        elif self.loss == 'influence':
            scores_batches = self._estimate_influence_if(model, al_datamodule, indices_batches)
        elif self.loss == 'gt':
            scores_batches = self._compute_gt_error(model, al_datamodule, indices_batches)
            scores_batches['gt'] = scores_batches['zero_one']
        elif self.loss == 'random':
            scores_batches = {'random': np.random.rand(len(indices_batches))}
        else:
            scores_batches = {}
        batch_features = self._get_batch_characteristics(model, al_datamodule, indices_batches)
        scores_batches.update(batch_features)

        avg_rank = np.mean([
            rankdata(batch_features['mmd']),
            rankdata(batch_features['uncertainty']),
            rankdata(-np.array(batch_features['density'])),
            rankdata(-np.array(batch_features['diversity'])),
        ], axis=0)
        scores_batches['avg_rank'] = avg_rank

        new_scores = np.array(scores_batches[self.loss])
        self.ema_scores = new_scores if self.ema_scores is None else \
            self.ema_lmb * self.ema_scores + (1 - self.ema_lmb) * new_scores
        scores_batches[f'{self.loss}_ema'] = self.ema_scores

        for key in scores_batches:
            self.all_scores[key].append(scores_batches[key])

        # self.init_scores = self.init_scores if hasattr(self, 'init_scores') else scores_batches[self.loss]
        if len(self.all_scores[self.loss]) > 1 and self.loss == 'zero_one':
            prev_scores = np.array(self.all_scores[self.loss][-2])
            curr_scores = np.array(self.all_scores[self.loss][-1])
            relative_change = (curr_scores - prev_scores) / prev_scores
            eer_change = np.abs(relative_change).max().item()
            self.history['eer_change'].append(eer_change)

        # Ranking
        ranking_batches = {k: rankdata(v) for k, v in scores_batches.items()}
        for key in ranking_batches:
            self.all_rankings[key].append(ranking_batches[key])

        if self.eval_gt or self.loss == 'gt':
            gt_scores_batches = self._compute_gt_error(model, al_datamodule, indices_batches)
            for key in gt_scores_batches:
                self.all_gt_scores[key].append(gt_scores_batches[key])
                self.all_gt_rankings[key].append(rankdata(gt_scores_batches[key]))

            gt_idx_best_batch = np.argmin(gt_scores_batches['zero_one'])
            print('Best batch was:', list(self.strategies.keys())[gt_idx_best_batch])
            spearmans = {key: spearmanr(scores_batches[key], gt_scores_batches['zero_one']).statistic.item()
                         for key in scores_batches}
            for key in spearmans:
                self.history[f'spearmans_{key}'].append(spearmans[key])
            self.history['best_strat'].append(gt_idx_best_batch.item())

        idx_best_batch = ranking_batches[f'{self.loss}_ema'].argmin()

        self.history['bought_strat'].append(idx_best_batch.item())
        print('Picked batch:', list(self.strategies.keys())[idx_best_batch])
        indices = indices_batches[idx_best_batch]
        return indices

    def _preselect_batches(self, model, al_datamodule, acq_size, subset_size, u_indices=None):
        u_indices = al_datamodule.unlabeled_indices if u_indices is None else u_indices
        num_batches_strats = {t: self.num_batches // len(self.strategies) for t in self.strategies}

        indices = []
        for strat_name, strat in track(self.strategies.items(), "Sampling candidate batches:"):
            num_batches = num_batches_strats[strat_name]

            indices_strat = []
            for i_batch in range(num_batches):
                aldm = copy.deepcopy(al_datamodule)

                sample_size = min(subset_size, len(u_indices))
                cand_indices = self.rng.choice(u_indices, size=sample_size, replace=False)
                aldm.unlabeled_indices = cand_indices.tolist()

                filter_acq_size = self.filter_acq_size if self.filter_acq_size is not None else acq_size
                idx = strat.query(model=model, al_datamodule=aldm, acq_size=filter_acq_size)
                indices_strat.append(idx)

            indices.extend(indices_strat)
        indices = np.array(indices)
        return indices

    def _get_batch_characteristics(self, model, al_datamodule, indices_batches):
        characteristics = defaultdict(list)
        output_types = ['logits', 'features']

        u_loader, _ = al_datamodule.unlabeled_dataloader()
        u_outputs = model.get_model_outputs(u_loader, output_types, device=self.device)
        l_loader, _ = al_datamodule.labeled_dataloader()
        l_outputs = model.get_model_outputs(l_loader, output_types, device=self.device)
        u_outputs['features_norm'] = F.normalize(u_outputs['features'], dim=-1)
        l_outputs['features_norm'] = F.normalize(l_outputs['features'], dim=-1)

        for batch_indices in indices_batches:
            b_loader = al_datamodule.custom_dataloader(indices=batch_indices.tolist())
            b_outputs = model.get_model_outputs(b_loader, output_types, device=self.device)
            b_outputs['features_norm'] = F.normalize(b_outputs['features'], dim=-1)
            # b_outputs['features'] = F.normalize(b_outputs['features'], dim=-1)

            from dal_toolbox.active_learning.strategies.uncertainty import margin_score
            logits = b_outputs['logits']
            avg_uncertainty = margin_score(logits.softmax(-1)).mean().item()

            features = b_outputs['features']
            dists = torch.cdist(features, features, p=2)
            avg_dist = dists.mean().item()

            u_features = u_outputs['features']
            dist = torch.cdist(features, u_features, p=2)
            knn_dists, _ = torch.topk(dist, k=20, dim=1, largest=False)
            avg_density = (1.0 / (1e-8 + knn_dists.mean(dim=1))).mean().item()

            l_features_norm = l_outputs['features_norm']
            b_features_norm = b_outputs['features_norm']
            u_features_norm = u_outputs['features_norm']
            new_l_features_norm = torch.cat((l_features_norm, b_features_norm))
            mmd = compute_mmd(new_l_features_norm, u_features_norm)
            mmd = mmd.item()

            characteristics['uncertainty'].append(avg_uncertainty)
            characteristics['density'].append(avg_density)
            characteristics['diversity'].append(avg_dist)
            characteristics['mmd'].append(mmd)

        return characteristics

    def _estimate_influence_fisher(self, model, al_datamodule, indices_batches, eps=1):
        scores_batches = defaultdict(list)
        l_indices = al_datamodule.labeled_indices

        if not isinstance(self.temp_scaling, bool):
            temp = float(self.temp_scaling)
            print(f'Scaling with temperature {temp}.')
        elif self.temp_scaling:
            temp = self._compute_temperature(model, al_datamodule)
            # temp = self._compute_gt_temperature(model, al_datamodule)
            print(f'Scaling with temperature {temp}.')
        else:
            temp = 1

        output_types = ['features', 'logits', 'mean_field_logits']
        val_loader = al_datamodule.custom_dataloader(self.val_indices)
        val_outputs = model.get_model_outputs(val_loader, output_types, device=self.device)
        val_logits = val_outputs['logits'] / temp
        val_features = val_outputs['features']

        A_val, B_val = fisher_AB_from_batch(val_logits, val_features, eps=1e-3)
        fisher_val_binary = fisher_binary(val_logits, val_features, eps=eps)
        # fisher_val = fisher_full(val_logits, val_features, eps=1e-3)

        for batch_indices in track(indices_batches, "Evaluating influence of candidate batches.."):
            new_indices = l_indices + batch_indices.tolist()
            b_loader = al_datamodule.custom_dataloader(indices=new_indices)

            b_outputs = model.get_model_outputs(b_loader, output_types, self.device)
            b_logits = b_outputs['logits'] / temp
            b_features = b_outputs['features']
            A_query, B_query = fisher_AB_from_batch(b_logits, b_features, eps=eps)

            t1 = torch.trace(torch.linalg.solve(A_query, A_val))
            t2 = torch.trace(torch.linalg.solve(B_query, B_val))
            fir_kfac = t1 * t2

            # fisher_query = fisher_full(b_logits, b_features, eps=eps)
            fisher_query_binary = fisher_binary(b_logits, b_features, eps=eps)
            fir_binary = torch.trace(torch.linalg.solve(fisher_query_binary, fisher_val_binary))

            # fisher_query = fisher_full(b_logits, b_features, eps=eps)
            # fir = torch.trace(torch.linalg.solve(fisher_query, fisher_val))

            # scores_batches['fir'].append(fir.item())
            scores_batches['fir_kfac'].append(fir_kfac.item())
            scores_batches['fir_binary'].append(fir_binary.item())
        return scores_batches

    def _estimate_influence_if(self, model, al_datamodule, indices_batches):
        scores_batches = defaultdict(list)
        val_loader = al_datamodule.custom_dataloader(self.val_indices)
        # temp = self._compute_temperature(model, al_datamodule)
        # trainer = Trainer(barebones=True, max_epochs=200)
        # trainer.fit(self.aux_model, al_datamodule.labeled_dataloader()[0])

        num_samples = 0
        params = [p for p in model.parameters()]
        val_grad = [torch.zeros_like(p) for p in params]
        for batch in val_loader:
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            with torch.enable_grad():
                logits = model(inputs)
                loss = F.cross_entropy(logits, logits.softmax(-1).detach()/.1, reduction='sum')
                grads = torch.autograd.grad(outputs=loss, inputs=params)
            for i, grad in enumerate(grads):
                val_grad[i] += grad.clone()
            num_samples += logits.size(0)
        val_grad = [grad / num_samples for grad in val_grad]

        @torch.enable_grad()
        def hessian_vector_product(model, s_test, inputs, labels, weight_decay):
            params = [param for param in model.parameters()]

            loss = F.cross_entropy(model(inputs), labels)
            reg = torch.cat([i.view(-1) ** 2 for i in params]).sum() / (2*inputs.size(0)) * weight_decay
            grad = torch.autograd.grad(outputs=loss + reg, inputs=params, create_graph=True)

            grad = torch.cat([item.view(-1) for item in grad])
            v_elem = torch.cat([item.view(-1) for item in s_test])
            # elemwise_product = [x * y for x, y in zip(grad, v_elem)]
            dot = torch.dot(grad, v_elem)
            elemgrad = torch.autograd.grad(outputs=dot, inputs=params, retain_graph=True)
            return elemgrad

        # Compute s_test
        r = 20
        recursion_depth = 50
        damping = 0.01
        scale = 25
        weight_decay = model.optimizer.param_groups[0]['weight_decay']

        l_loader, _ = al_datamodule.labeled_dataloader()
        final_s_test = []
        for i in track(range(r)):
            s_test = [item.clone() for item in val_grad]
            for ep in range(recursion_depth):
                for batch in l_loader:
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    hessian_vector_val = hessian_vector_product(model, s_test, inputs, labels, weight_decay)
                    s_test = [a + (1 - damping) * b - c / scale for (a, b, c)
                              in zip(val_grad, s_test, hessian_vector_val)]
            final_s_test.append(s_test)
        flat_s = [torch.cat([p.reshape(-1) for p in s]) for s in final_s_test]
        hv = torch.stack(flat_s).mean(0)

        for batch_indices in indices_batches:
            b_loader = al_datamodule.custom_dataloader(batch_indices.tolist(), custom_batch_size=1)

            exp_influences = []
            for batch in b_loader:
                input = batch[0].to(self.device)
                # model.set_mean_field_factor(.01)
                influences = torch.zeros(10, device=self.device)
                for lbl in torch.arange(10, device=self.device):
                    with torch.enable_grad():
                        logits = model(input)
                        loss = F.cross_entropy(logits, lbl.view(1), reduction='sum')
                        grad = torch.autograd.grad(outputs=loss, inputs=params)
                    grad = torch.cat([g.view(-1) for g in grad])
                    influence = - torch.matmul(grad, hv)
                    influences[lbl] = influence
                probas = model(input).softmax(-1).view(-1)
                exp_influences.append(influences[probas.argmax()])
                # exp_influences.append(torch.sum(probas*influences))
                # exp_influences.append(influences[batch[1]])
            batch_influence = torch.stack(exp_influences).mean().item()
            scores_batches['influence'].append(batch_influence)
        return scores_batches

    def _estimate_expected_error(self, model, al_datamodule, indices_batches):
        scores_batches = defaultdict(list)

        if self.perf_estimation == 'unlabeled_pool' or self.perf_estimation == 'noisy_label_dataset':
            val_loader = al_datamodule.custom_dataloader(self.val_indices)
        elif self.perf_estimation == 'labeled_pool':
            val_loader = l_loader
        elif self.perf_estimation == 'labeled_val':
            val_loader = al_datamodule.val_dataloader()
        else:
            raise NotImplementedError(f'Performance estimation {self.perf_estimation} not implemented.')

        if not isinstance(self.temp_scaling, bool):
            temp = float(self.temp_scaling)
        elif self.temp_scaling:
            temp = self._compute_temperature(model, al_datamodule)
            # temp = self._compute_gt_temperature(model, al_datamodule)
            print(f'Scaling with temperature {temp}.')
        else:
            temp = 1

        l_loader, l_indices = al_datamodule.labeled_dataloader()
        output_types = ['logits', 'mean_field_logits', 'features', 'labels']
        l_outputs = model.get_model_outputs(l_loader, output_types=output_types, device=self.device)
        val_outputs = model.get_model_outputs(val_loader, output_types, device=self.device)

        base_model = copy.deepcopy(model)
        for batch_indices in track(indices_batches, "Evaluating influence of candidate batches.."):
            b_loader = al_datamodule.custom_dataloader(batch_indices, train=False)
            b_outputs = base_model.get_model_outputs(b_loader, output_types, device=self.device)
            b_outputs['logits'] = b_outputs['logits'] / temp
            if self.look_ahead == 'pseudo_labels':
                labels = b_outputs['logits'].argmax(dim=-1).unsqueeze(0)
            elif self.look_ahead == 'mc_labels':  # Samples labels via Monte Carlo
                categorical = torch.distributions.Categorical(logits=b_outputs['logits'])
                labels = categorical.sample((self.num_mc_labels,))
            elif self.look_ahead == 'soft_labels':
                labels = (b_outputs['logits']).softmax(-1).unsqueeze(0)
                l_labels_onehot = F.one_hot(l_outputs['labels'], num_classes=b_outputs['logits'].size(-1))
            elif self.look_ahead == 'noise_labels':  # Artificially change a percentage of labels to be wrong
                labels = b_outputs['labels']
                num_wrong = int(len(labels)*self.label_noise_rate)
                rnd_idx = torch.randperm(len(labels))[:num_wrong]
                labels[rnd_idx] = torch.randint(0, b_outputs['logits'].size(-1), (num_wrong,))
                labels = labels.unsqueeze(0)
            elif self.look_ahead == 'label_prop':
                X_lp = torch.cat((l_outputs['features'], val_outputs['features'], b_outputs['features']))
                y_lp = torch.full((len(X_lp),), -1)
                y_lp[:len(l_indices)] = l_outputs['labels']

                lp = LabelSpreading(kernel='knn', n_neighbors=20)
                X_lp = StandardScaler().fit_transform(X_lp)
                lp.fit(X_lp, y_lp)
                labels = lp.transduction_[-len(batch_indices):]
                labels = torch.from_numpy(labels).unsqueeze(0)
                val_labels = lp.transduction_[len(l_indices):-len(batch_indices)]
                val_labels = torch.from_numpy(val_labels)
            elif self.look_ahead == 'true_labels':
                labels = b_outputs['labels'].unsqueeze(0)
            else:
                raise NotImplementedError(f'Look-ahead strategy {self.look_ahead} not implemented.')

            l_labels_ = l_outputs['labels'] if self.look_ahead != 'soft_labels' else l_labels_onehot
            loss_labels = defaultdict(list)
            for i_labels, labels_batch in enumerate(labels):
                retrain_indices = l_indices + batch_indices.tolist()
                custom_labels = torch.cat((l_labels_, labels_batch))
                retrain_loader = al_datamodule.custom_dataloader(
                    indices=retrain_indices, train=True, custom_labels=custom_labels)
                model.reset_states()
                trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
                trainer.fit(model, retrain_loader)

                # Evaluate the model after retraining
                val_outputs_re = model.get_model_outputs(val_loader, output_types, device=self.device)
                if self.perf_estimation == 'labeled_val':
                    logits = val_outputs_re['logits']
                    targets = val_outputs_re['labels']
                    ce = F.cross_entropy(logits, targets)
                    accuracy = (val_outputs_re['logits'].argmax(-1) ==
                                val_outputs_re['labels']).float().mean()
                else:
                    logits = val_outputs_re['logits']
                    targets = logits.softmax(-1)
                    ce = F.cross_entropy(logits, targets)
                    accuracy = torch.mean(logits.softmax(-1).max(-1).values)
                    # accuracy = logits.softmax(-1).gather(1, val_labels.view(-1, 1)).mean()

                probas_before = val_outputs['logits'].softmax(-1)
                probas_after = val_outputs_re['logits'].softmax(-1)
                moc = torch.norm(probas_before - probas_after, dim=-1).sum(-1)
                kl = F.kl_div(probas_after.log(), probas_before, reduction='batchmean')
                zero_one = 1 - accuracy

                probas_before = val_outputs['mean_field_logits'].softmax(-1)
                probas_after = val_outputs_re['mean_field_logits'].softmax(-1)
                mocu = torch.norm(probas_before - probas_after, dim=-1).sum(-1)

                loss_labels['cross_entropy'].append(ce.item())
                loss_labels['zero_one'].append(zero_one.item())
                loss_labels['moc'].append(-moc.item())
                loss_labels['mocu'].append(-mocu.item())
                loss_labels['kl'].append(-kl.item())

            for key in loss_labels:
                scores_batches[key].append(np.mean(loss_labels[key]).item())
        return scores_batches

    def _compute_gt_error(self, model, al_datamodule, indices_batches):
        l_indices = al_datamodule.labeled_indices
        gt_val_loader = al_datamodule.val_dataloader()

        # Computes the ground truth loss per batch for testing
        gt_loss_batches = defaultdict(list)
        for batch_indices in track(indices_batches, 'Evaluating GT influence of candidate batches'):
            retrain_indices = l_indices + batch_indices.tolist()
            gt_retrain_loader = al_datamodule.custom_dataloader(indices=retrain_indices, train=True)
            model.reset_states(reset_model_parameters=True)
            trainer = Trainer(barebones=True, max_epochs=200)
            trainer.fit(model, gt_retrain_loader)
            val_outputs = model.get_model_outputs(gt_val_loader, ['logits', 'labels'], device=self.device)
            gt_ce = F.cross_entropy(val_outputs['logits'], val_outputs['labels'])
            gt_zo = (val_outputs['logits'].argmax(-1) != val_outputs['labels']).float().mean()
            gt_loss_batches['cross_entropy'].append(gt_ce.item())
            gt_loss_batches['zero_one'].append(gt_zo.item())
        return gt_loss_batches

    def _compute_temperature(self, model, al_datamodule, num_splits=10, num_train_epochs=200, candidate_temps=None):
        if candidate_temps is None:
            candidate_temps = np.arange(.5, 10, 0.1)

        l_indices = al_datamodule.labeled_indices
        kfold = KFold(n_splits=num_splits)
        model = copy.deepcopy(model)
        logits, labels = [], []
        for train_indices, test_indices in kfold.split(l_indices):
            train_loader = al_datamodule.custom_dataloader(train_indices, train=True)
            trainer = Trainer(barebones=True, max_epochs=num_train_epochs)
            model.reset_states(reset_model_parameters=True)
            trainer.fit(model, train_loader)

            # val_loader = al_datamodule.custom_dataloader(test_indices, train=False)
            val_loader = al_datamodule.custom_dataloader(test_indices, train=False)
            val_outputs = model.get_model_outputs(val_loader, ['logits', 'labels'], device=self.device)
            logits.append(val_outputs['logits'])
            labels.append(val_outputs['labels'])
        logits = torch.cat(logits)
        labels = torch.cat(labels)

        best_ece = float('inf')
        best_temp = 1.0
        for temp in candidate_temps:
            scaled_logits = logits / temp
            ece = F.cross_entropy(scaled_logits, labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return best_temp.item()

    def _compute_gt_temperature(self, model, al_datamodule, num_splits=10, num_train_epochs=200, candidate_temps=None):
        if candidate_temps is None:
            candidate_temps = np.arange(.5, 10, 0.1)

        model = copy.deepcopy(model)
        model.reset_states(reset_model_parameters=True)
        train_loader = al_datamodule.custom_dataloader(al_datamodule.labeled_indices, train=True)
        Trainer(barebones=True, max_epochs=num_train_epochs).fit(model, train_loader)
        val_loader = al_datamodule.custom_dataloader(al_datamodule.unlabeled_indices, train=False)
        val_outputs = model.get_model_outputs(val_loader, ['logits', 'labels'], device=self.device)

        logits = val_outputs['logits']
        labels = val_outputs['labels']
        best_ece = float('inf')
        best_temp = 1.0
        for temp in candidate_temps:
            scaled_logits = logits / temp
            ece = self.compute_ece(scaled_logits, labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return best_temp.item()

    def _estimate_expected_error_cv(self, model, al_datamodule, indices_batches):
        scores_batches = defaultdict(list)
        output_types = ['logits', 'mean_field_logits', 'features', 'labels']

        l_loader, l_indices = al_datamodule.labeled_dataloader()
        l_outputs = model.get_model_outputs(l_loader, output_types=output_types, device=self.device)
        base_model = copy.deepcopy(model)

        for batch_indices in track(indices_batches, "Evaluating influence of candidate batches.."):
            b_loader = al_datamodule.custom_dataloader(batch_indices, train=False)
            b_outputs = base_model.get_model_outputs(b_loader, output_types, device=self.device)

            pseudo_labels = b_outputs['logits'].argmax(-1)
            l_labels = l_outputs['labels']

            new_l_indices = torch.Tensor(l_indices + batch_indices.tolist()).long()
            new_l_labels = torch.cat((l_labels, pseudo_labels))

            scores_cv = defaultdict(list)
            kfold = KFold(shuffle=True)
            for train_idx, val_idx in kfold.split(new_l_indices):
                train_indices = new_l_indices[train_idx]
                train_labels = new_l_labels[train_idx]
                train_loader = al_datamodule.custom_dataloader(
                    train_indices, custom_labels=train_labels, train=True)
                model.reset_states()
                trainer = Trainer(barebones=True, max_epochs=self.num_retraining_epochs)
                trainer.fit(model, train_loader)

                val_indices = new_l_indices[val_idx]
                val_labels = new_l_labels[val_idx]
                val_loader = al_datamodule.custom_dataloader(
                    val_indices, custom_labels=val_labels, train=False)
                val_outputs = model.get_model_outputs(val_loader, output_types, device=self.device)

                val_logits, val_targets = val_outputs['logits'], val_outputs['labels']
                cross_entropy = F.cross_entropy(val_logits, val_targets)
                accuracy = (val_logits.argmax(-1) == val_targets).float().mean()
                zero_one = 1 - accuracy
                scores_cv['cross_entropy'].append(cross_entropy.item())
                scores_cv['zero_one'].append(zero_one.item())
            for key in scores_cv:
                scores_batches[key].append(np.mean(scores_cv[key]).item())
        return scores_batches


class TCM(Query):
    def __init__(self, typi_steps=3, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.typi_steps = typi_steps

        self.typiclust = strategies.TypiClust(subset_size=subset_size, device=device)
        self.margin = strategies.MarginSampling(subset_size=subset_size, device=device)
        self.iter = 0

    def query(self, *, model, al_datamodule, acq_size):
        if self.iter < self.typi_steps:
            indices = self.typiclust.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
        else:
            indices = self.margin.query(model=model, al_datamodule=al_datamodule, acq_size=acq_size)
        self.iter += 1
        return indices


class SelectAL(Query):
    def __init__(self,
                 epsilon=.05,
                 low_budget_strategy='typiclust',
                 high_budget_strategy='badge',
                 surrogate_low_strategy='typiclust',
                 surrogate_high_strategy='inv_typiclust',
                 num_random_reps=5,
                 val_split=0.05,
                 num_val_reps=5,
                 train_epochs=50,
                 subset_size=None,
                 random_seed=None,
                 device='cpu',
                 ):
        super().__init__(random_seed)
        self.epsilon = epsilon
        self.subset_size = subset_size
        self.device = device
        self.num_random_reps = num_random_reps
        self.num_val_reps = num_val_reps
        self.val_split = val_split
        self.train_epochs = train_epochs
        self.history = []

        surrogate_strat_dict = build_al_strategies(
            ['random', surrogate_low_strategy, surrogate_high_strategy], device=self.device)
        self.random_strategy = surrogate_strat_dict['random']
        self.surrogate_low_strategy = surrogate_strat_dict[surrogate_low_strategy]
        self.surrogate_high_strategy = surrogate_strat_dict[surrogate_high_strategy]
        self.surrogate_strategies = [
            self.random_strategy,
            self.surrogate_low_strategy,
            self.surrogate_high_strategy
        ]

        strat_dict = build_al_strategies(
            ['random', low_budget_strategy, high_budget_strategy], device=self.device)
        self.low_budget_strategy = strat_dict[low_budget_strategy]
        self.high_budget_strategy = strat_dict[high_budget_strategy]
        self.strategies = [
            self.random_strategy,
            self.low_budget_strategy,
            self.high_budget_strategy
        ]
        self.query_history = []

    def query(self, *, model, al_datamodule, acq_size):
        al_datamodule = copy.deepcopy(al_datamodule)
        _, labeled_indices = al_datamodule.labeled_dataloader()
        if self.epsilon >= len(labeled_indices):
            raise ValueError(
                f'Epsilon={self.epsilon} greater or equals the labeled pool size of {len(labeled_indices)}.')

        # Determine the regime we are in
        labeled_labels = torch.cat([batch[1] for batch in al_datamodule.custom_dataloader(labeled_indices)])
        labels_unique, labels_counts = labeled_labels.unique(return_counts=True)
        num_classes = len(labels_unique)

        min_lbl_count = labels_counts.min().item()
        eps = self.epsilon if not (0 < self.epsilon < 1) else int(self.epsilon*len(labeled_indices))
        c = max(eps // num_classes, 1)
        c = min(c, min_lbl_count)

        surrogate_accs = []
        for i_strat, strat in enumerate(self.surrogate_strategies):
            num_reps = 1 if i_strat != 0 else self.num_random_reps

            rep_accs = []
            for _ in range(num_reps):
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
                    if len(new_indices) > 2: # NOTE: This prevents issues for performance eval below
                        new_indices.remove(idx)

                # Eval the performance via cross validation on new labeled data
                accs = []
                num_val_samples = max(1, int(len(new_indices)*self.val_split))
                for _ in range(self.num_val_reps):
                    val_indices = self.rng.choice(new_indices, size=num_val_samples, replace=False)
                    train_indices = np.setdiff1d(new_indices, val_indices)
                    model = self.train_model(model, aldm.custom_dataloader(train_indices, train=True))
                    acc = self.evaluate_model(model, aldm.custom_dataloader(val_indices)).item()
                    accs.append(acc)
                rep_accs.append(np.mean(accs))
            surrogate_accs.append(np.mean(rep_accs).item())
        idx = self.rng.choice(np.flatnonzero(surrogate_accs == np.min(surrogate_accs)))

        selected = {k: v.item() for k, v in zip(['random', 'low', 'high'], np.eye(3, dtype=int)[idx])}
        self.query_history.append(selected)

        strat = self.strategies[idx]
        if self.subset_size is not None:
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


class TAILOR(Query):
    def __init__(
            self,
            num_classes=None,
            subset_size=None,
            random_seed=None,
            al_strategies=['random', 'margin', 'badge', 'bait', 'typiclust',
                            'alfamix', 'dropquery', 'max_herding', 'unc_herding'],
            device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device
        self.last_w = None
        self.num_classes = num_classes
        self.strategies = list(build_al_strategies(al_strategies=al_strategies,
                               device=self.device, subset_size=subset_size).values())
        self.stats = np.ones((2, len(self.strategies), self.num_classes)) if num_classes is not None else None

    def query(self, *, model, al_datamodule, acq_size):
        al_datamodule = copy.deepcopy(al_datamodule)
        if self.stats is None:
            dl = al_datamodule.custom_dataloader([0])
            logits = model.get_model_outputs(dl, ['logits'])['logits']
            self.num_classes = logits.size(-1)
            self.stats = np.ones((2, len(self.strategies), self.num_classes))

        # Retrieve a one-hot encoding of current labels
        l_loader, _ = al_datamodule.labeled_dataloader()
        l_outputs = model.get_model_outputs(l_loader, output_types=['labels'], device=self.device)
        l_labels = torch.nn.functional.one_hot(l_outputs['labels'], num_classes=self.num_classes).numpy()

        # This is the div-metric of the original paper for multi-class classification
        # Multi-Label Criterion is omitted for the sake of clarity.
        w = np.sum(l_labels, axis=0)
        w = np.clip(w, a_min=1, a_max=None)
        w[(w / float(l_labels.shape[0])) > .5] *= -1
        v = 1 / w

        queried_indices = []
        new_stats = np.zeros_like(self.stats)
        for _ in range(acq_size):
            # Calculate expected reward
            theta = np.array([self.rng.dirichlet(self.stats[0, i, :] + 1)
                             for i in range(len(self.strategies))])
            expected_reward = theta @ v

            # Select Algorithm which maximizes expected reward and let it sample one sample
            strat_idx = np.argmax(expected_reward).item()
            # FIXME: Issues with TypiClust on TinyImageNet...
            queried_idx = self.strategies[strat_idx].query(
                model=model, al_datamodule=al_datamodule, acq_size=1)[0]
            queried_indices.append(queried_idx)

            # Update al_datamodule to exclude this index from subsequent querying
            al_datamodule.unlabeled_indices.remove(queried_idx)

            # Update stats in new_stats. Note that these new labels are not used during querying but only
            # afterwards to update self.stats, which is used for strategy selection.
            label = al_datamodule.train_dataset[queried_idx][1]
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).numpy()
            new_stats[0, strat_idx] += label
            new_stats[1, strat_idx] += 1 - label

        # As if labels are only revealed now.
        self.stats = self.stats * .9 + new_stats
        return queried_indices


class AutoAL(Query):
    def __init__(self,
                 args,
                 al_strategies=['random', 'margin', 'badge', 'bait', 'typiclust',
                                'alfamix', 'dropquery', 'max_herding', 'unc_herding'],
                 subset_size=None,
                 device='cpu',
                 random_seed=None,
                 feature_dim=384,
                 num_classes=10,
                 ratio=0.02,
                 num_bilevel_epochs=50, #NOTE: Reduced from 400 as we dont fit full ResNet18
                 ):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device
        self.num_classes = num_classes
        self.embed_dim = feature_dim
        self.num_strats = len(al_strategies)
        self.num_bilevel_epochs = num_bilevel_epochs
        self.ratio = ratio

        # Additional HPs that can be calculated based on AL args
        self.num_acq = args.dataset.num_acq
        self.num_init = args.dataset.num_init
        self.acq_size = args.dataset.acq_size
        self.quota = self.num_init + self.num_acq * self.acq_size

        # Define SearchNet and FitNet as two laplace models
        self.search_net = build_model(args=args, num_features=self.embed_dim, num_classes=self.num_strats)
        self.fit_net = build_model(args=args, num_features=self.embed_dim, num_classes=self.num_classes)

        # Build Query Strategies required for AutoAL
        self.strategies = build_al_strategies(al_strategies, device=self.device, subset_size=subset_size)
        self.args = args


    # Removed the torch.no_grad context as stuff needs to be learned in here
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        al_datamodule = copy.deepcopy(al_datamodule)
        _, labeled_indices = al_datamodule.labeled_dataloader()

        # Split available labeled samples into training and validation data
        # Note: Custom Val Loader because we need DropLast to avoid small batches
        train_indices = random.sample(labeled_indices, k=len(labeled_indices)//2)
        val_indices = [i for i in labeled_indices if i not in train_indices]
        train_loader = DataLoader(dataset=Subset(al_datamodule.query_dataset, indices=train_indices), 
                                batch_size=al_datamodule.train_batch_size, 
                                drop_last=(len(train_indices)>al_datamodule.train_batch_size),
                                shuffle=True)
        
        # First, fit the fitnet on half of the labeled pool to have a solid foundation
        val_loader = al_datamodule.custom_dataloader(indices=val_indices, train=True)
        trainer = Trainer(max_epochs=self.args.model.num_epochs, barebones=True)
        trainer.fit(self.fit_net, val_loader)

        # Next, perform Bilevel optimization to fit both SearchNet and FitNet
        self.bilevel_optimization(model, train_loader, al_datamodule)

        #'accuracy': 0.928        eben sogar 0.931 bei 50 epochen    

        # Begin actual selection of samples to annotate by calculating scores for the 
        # unlabeled samples and take the topk
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        with torch.no_grad():
            _, _, _, new_score, _ = self.compute_scores(model=model,
                                                              dataloader=unlabeled_dataloader,
                                                              dataloader_inidces=unlabeled_indices,
                                                              al_datamodule=al_datamodule)
        new_score = torch.from_numpy(np.abs(new_score))
        _, local_indices = new_score.topk(k=acq_size)
        global_indices = [unlabeled_indices[idx] for idx in local_indices]
        return global_indices
    

    def get_outputs(self, dataloader, net_type):
        outputs = defaultdict(list)
        for batch in dataloader:
            inputs = batch[0].to(self.device)
            labels = batch[1]

            if net_type == 'fit_net':
                self.fit_net.to(self.device)
                features = self.fit_net.model.forward_features(inputs)
                logits = self.fit_net.model.forward_head(features)
                outputs['logits'].append(logits.cpu())
                outputs['labels'].append(labels)
            elif net_type == 'search_net':
                self.search_net.to(self.device)
                features = self.search_net.model.forward_features(inputs)
                logits = self.search_net.model.forward_head(features)
                outputs['logits'].append(logits.cpu())
                outputs['features'].append(features.cpu())
            else:
                raise AssertionError()

        outputs = {key: torch.cat(val) if isinstance(
            val[0], torch.Tensor) else val for key, val in outputs.items()}
        return outputs


    def get_score_matrix(self, model, al_datamodule, dataloader_indices):
        # ACQ Size needs to be adjusted depening on dataloader size
        if len(al_datamodule.unlabeled_indices) > (self.quota + self.num_init):
            num_q = self.acq_size
        else:
            if math.ceil(len(al_datamodule.unlabeled_indices) * self.ratio) == 1:
                num_q = 2            
            else:
                num_q = math.ceil(len(al_datamodule.unlabeled_indices) * self.ratio)

        # Create and fill the binary score matrix (num_samples, num_strategies)
        binary_matrix = np.zeros(shape=(len(al_datamodule.unlabeled_indices), len(self.strategies)))
        for i, strat in enumerate(self.strategies):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                queried_indices = self.strategies[strat].query(model=model, al_datamodule=al_datamodule, acq_size=num_q)
            for idx in dataloader_indices:
                if idx in queried_indices:
                    binary_matrix[dataloader_indices.index(idx)][i] = 1
        return binary_matrix
    

    def compute_scores(self, model, dataloader, dataloader_inidces, al_datamodule):
        # Get logits and score matrix for unlabeled samples
        output_search = self.get_outputs(dataloader, net_type="search_net")
        features_search, logits_search = output_search['features'], output_search['logits']

        # Adapt ALDatamodule so the query strategies can evaluate the custom made "unlabeled pool"
        aldm_copy = copy.deepcopy(al_datamodule)
        aldm_copy.unlabeled_indices = dataloader_inidces
        score_matrix = self.get_score_matrix(model, aldm_copy, dataloader_inidces)

        # Calculate various scores (see explanation above)
        weighted_score_matrix = torch.from_numpy(score_matrix).requires_grad_() * logits_search # (N_Samples, N_Strategies) 
        average_scores = weighted_score_matrix.sum(dim=1).softmax(dim=0) # (N_Samples)
        detached_scores = average_scores.cpu().detach().numpy() # (N_Samples)

        # Probably the "Sigmoid Relaxation they talked about"
        gmm = GaussianMixture(n_components=min(5, detached_scores.shape[0]))
        gmm.fit(detached_scores.reshape(-1, 1))
        confidence_interval = gmm.sample(n_samples=10000)[0]
        interval_max = np.percentile(confidence_interval, 100 - self.ratio * 100)
        relaxed_scores = torch.sigmoid(100000000000 * (average_scores - interval_max))

        # Retrieve predictions for fitnet
        output_fit = self.get_outputs(dataloader, net_type="fit_net")
        logits_fit, labels_fit = output_fit['logits'], output_fit['labels'] 

        return relaxed_scores, logits_fit, labels_fit, detached_scores, features_search
    

    def bilevel_optimization(self, model, dataloader, al_datamodule):
        # Initialize optimizer and lr-scheduler for Bilevel optimization
        self.search_net.train()
        optimizer_search = torch.optim.Adam(params=self.search_net.parameters(), weight_decay=5e-4)
        scheduler_search = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_search, T_max=self.num_bilevel_epochs//2)

        self.fit_net.train()
        optimizer_fit = torch.optim.Adam(params=self.fit_net.parameters(), weight_decay=5e-4)
        scheduler_fit = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_fit, T_max=self.num_bilevel_epochs//2)

        for i_optim_step in range(self.num_bilevel_epochs):
            input_search, target_search, new_dataloader_indices = next(dataloader.__iter__())
            subset = TensorDataset(input_search, target_search)
            one_batch_dataloader = DataLoader(subset, batch_size=input_search.size(0), shuffle=True)
            score, logits_fit, labels_fit, _, _ = self.compute_scores(model=model,
                                                                    dataloader=one_batch_dataloader,
                                                                    dataloader_inidces=new_dataloader_indices.tolist(),
                                                                    al_datamodule=al_datamodule
                                                                    )
            loss_fit = F.cross_entropy(logits_fit, labels_fit, reduction='none')
            # 1: Optimize search net
            if i_optim_step % 2 == 1:
                optimizer_search.zero_grad()
                loss_fit = loss_fit.detach() # Only search net should be updated

                # Final Loss is then a scalar product of predicted loss (score) and actual loss
                total_loss = -(loss_fit * score).mean()
                total_loss.backward()
                optimizer_search.step()
                scheduler_search.step()

            # 2: Optimize FitNet
            else:
                optimizer_fit.zero_grad()
                score = score.detach() # Only fitnet should be updated

                # Final Loss is, again, a scalar product of predicted loss (score) and actual loss
                total_loss = (loss_fit * score).mean()
                total_loss.backward()
                optimizer_fit.step()
                scheduler_fit.step()



def build_al_strategies(al_strategies, device='cpu', subset_size=None):
    strat_dict = {
        'random': lambda: strategies.RandomSampling(),
        'margin': lambda: strategies.MarginSampling(device=device, subset_size=subset_size),
        'coreset': lambda: strategies.CoreSet(device=device, subset_size=subset_size),
        'badge': lambda: strategies.Badge(device=device, subset_size=subset_size),
        'typiclust': lambda: strategies.TypiClust(device=device, subset_size=subset_size),
        'inv_typiclust': lambda: strategies.InverseTypiClust(device=device, subset_size=subset_size),
        'alfamix': lambda: strategies.AlfaMix(device=device, subset_size=subset_size),
        'dropquery': lambda: strategies.DropQuery(device=device, subset_size=subset_size),
        'bait': lambda: strategies.BaitSampling(grad_likelihood='binary_cross_entropy', device=device, subset_size=subset_size),
        'max_herding': lambda: strategies.MaxHerding(device=device, subset_size=subset_size),
        'unc_herding': lambda: strategies.UncertaintyHerding(device=device, subset_size=subset_size),
        'coreset': lambda: strategies.CoreSet(subset_size=subset_size, device=device),
    }
    strategies_dict = {}
    for strat_name in al_strategies:
        if strat_name not in strat_dict:
            raise NotImplementedError(f"Strategy '{strat_name}' not implemented.")
        strategies_dict[strat_name] = strat_dict[strat_name]()
    return strategies_dict


def sobol_mc_labels(logits, num_mc_labels):
    batch_size, num_classes = logits.shape
    probas = torch.softmax(logits, dim=-1).cpu().numpy()
    sampler = qmc.Sobol(d=batch_size, scramble=True)
    sobol_samples = sampler.random(num_mc_labels)
    labels = []
    for i in range(batch_size):
        cdf = probas[i].cumsum()
        label_indices = (sobol_samples[:, i][:, None] < cdf).argmax(axis=1)
        labels.append(label_indices)
    labels = np.stack(labels, axis=1)
    return torch.from_numpy(labels).long()


class EMACallback(Callback):
    def __init__(self, decay=0.999, start_step=100):
        self.decay = decay
        self.ema_model = None
        self.start_step = start_step

    def on_fit_start(self, trainer, pl_module):
        self.ema_model = AveragedModel(pl_module, avg_fn=get_ema_avg_fn(self.decay))

    def on_train_batch_end(self, trainer, pl_module, *args):
        if trainer.global_step > self.start_step:
            self.ema_model.update_parameters(pl_module)

    def on_fit_end(self, trainer, pl_module):
        # overwrite the training model with EMA weights
        pl_module.model.load_state_dict(self.ema_model.module.model.state_dict())


def get_ema_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across a single param."""
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param
    return ema_update


@torch.no_grad()
def fisher_binary(logits, features, eps=1e-2):
    device = logits.device
    grad_repr = get_exp_grad_representations(
        logits, features, grad_likelihood='binary_cross_entropy', device=device)
    fisher = torch.zeros((grad_repr.size(-1), grad_repr.size(-1)), device=device)
    for repr in torch.utils.data.DataLoader(grad_repr, batch_size=10, shuffle=False):
        fisher += torch.matmul(repr.transpose(1, 2), repr).sum(0)
    fisher /= grad_repr.size(0)
    fisher += eps*torch.eye(len(fisher), device=device)
    return fisher.cpu()


def fisher_full(logits, features, eps=1e-2):
    device = logits.device
    num_classes = logits.size(-1)
    num_features = features.size(-1)
    device = 'cuda'

    probas = logits.softmax(-1)
    fisher = torch.zeros(num_classes*num_features, num_classes*num_features, device=device)
    for proba, feature in list(zip(probas, features)):
        proba = proba.to(device)
        feature = feature.to(device)
        t1 = torch.diag(proba) - torch.outer(proba, proba)
        t2 = torch.outer(feature, feature)
        fisher += torch.kron(t1, t2)
    fisher += eps*torch.eye(len(fisher), device=device)
    return fisher.cpu()


@torch.no_grad()
def fisher_AB_from_batch(logits, features, eps=1e-2):
    probas = torch.softmax(logits, dim=-1)
    probas_mean = probas.mean(dim=0)
    probas_cov = (probas.T @ probas) / probas.size(0)
    A = torch.diag(probas_mean) - probas_cov

    B = (features.T @ features) / features.size(0)

    A = A + eps * torch.eye(A.size(0), device=A.device)
    B = B + eps * torch.eye(B.size(0), device=B.device)
    return A, B


class LossTracker(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(loss.detach().cpu().item())


def compute_mmd(X, Y, gamma=None):
    """
    Compute unbiased MMD^2 between two sets X and Y with an RBF kernel.

    Args:
        X: np.array of shape [n, d]
        Y: np.array of shape [m, d]
        gamma: RBF kernel bandwidth (if None, set to 1 / d)

    Returns:
        mmd2: float, squared MMD value
    """
    if gamma is None:
        l_dist = pairwise_distances(X, X)
        np.fill_diagonal(l_dist, np.inf)
        min_dist = l_dist.min()
        np.where(l_dist == 0)
        gamma = 1.0 / (min_dist**2) if min_dist > 0 else 1.0

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    m = X.shape[0]
    n = Y.shape[0]

    # Unbiased estimate (leave out diagonal terms)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) \
        + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) \
        - 2 * Kxy.mean()

    return mmd2
