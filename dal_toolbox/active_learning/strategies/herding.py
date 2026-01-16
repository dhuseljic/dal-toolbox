import warnings
import torch
import copy
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_kernels, pairwise_distances
from lightning import Trainer

from . import Query
from ..data import ActiveLearningDataModule
from ...metrics.calibration import ExpectedCalibrationError


class UncertaintyHerding(Query):
    def __init__(self, kernel='rbf', uncertainty='margin', subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device
        self.uncertainty = uncertainty
        self.compute_ece = ExpectedCalibrationError()
        self.kernel = kernel
        self.history = defaultdict(list)
        self.cycle = None
        

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        u_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        l_loader, _ = al_datamodule.labeled_dataloader()

        u_outputs = model.get_model_outputs(u_loader, output_types=['logits', 'features'], device=self.device)
        l_outputs = model.get_model_outputs(l_loader, output_types=['features'], device=self.device)

        u_logits = u_outputs['logits']
        u_features = u_outputs['features']
        l_features = l_outputs.get('features', torch.empty(0, device=self.device))

        # Compute temperature
        if self.cycle != None and len(self.history["temperature"]) == self.cycle:
            temperature = self.history["temperature"][-1]
        else:
            temperature = self._compute_temperature(model, al_datamodule)
            self.history['temperature'].append(temperature.item())
        
        # Calibrate the model via temperature scaling
        u_logits = u_logits / temperature
        uncertainty_scores = self._compute_uncertainty_scores(u_logits)
        uncertainty_scores = uncertainty_scores.reshape(-1, 1).numpy()
        
        # Compute kernel matrices
        all_features = torch.cat((u_features, l_features), dim=0)
        all_features = self._normalize_features(all_features)
        u_features = all_features[:len(u_features)].cpu().numpy()
        l_features = all_features[len(u_features):].cpu().numpy()

        # Compute min dist
        if self.cycle != None and len(self.history["min_dist"]) == self.cycle:
            min_dist = self.history["min_dist"][-1]
        else:
            l_dist = pairwise_distances(l_features, l_features)
            np.fill_diagonal(l_dist, np.inf)
            min_dist = l_dist.min()
            self.history['min_dist'].append(min_dist.item())

        # https://github.com/BorealisAI/uherding/blob/main/deep-al/pycls/al/uherding.py#L90
        gamma = 1.0 / (min_dist**2) if min_dist > 0 else 1.0
        def kernel_func(x, y): return pairwise_kernels(x, y, metric=self.kernel, gamma=gamma)
        
        if len(l_features) != 0:
            K_ = kernel_func(u_features, l_features)
            max_kernels = K_.max(axis=1, keepdims=True)
        else:
            max_kernels = np.zeros((len(u_features), 1))
        K = kernel_func(u_features, u_features)

        selected = []
        for _ in range(acq_size):
            idx = np.argmax(np.mean(uncertainty_scores*np.maximum(K - max_kernels, 0), axis=1))
            max_kernels = np.maximum(max_kernels, kernel_func(u_features, u_features[np.newaxis, idx]))
            selected.append(idx)

        return [unlabeled_indices[idx] for idx in selected]

    def _compute_temperature(self, model, al_datamodule, val_size=0.2, num_train_epochs=200, candidate_temps=None):
        if candidate_temps is None:
            # https://github.com/BorealisAI/uherding/blob/main/deep-al/pycls/calibration/temperature_scaling.py#L17
            candidate_temps = np.arange(1.0, 20, 0.1)
        l_indices = al_datamodule.labeled_indices
        train_idx, val_idx = train_test_split(l_indices, test_size=val_size, random_state=self.random_seed)

        train_loader = al_datamodule.custom_dataloader(train_idx, train=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(barebones=True, max_epochs=num_train_epochs)
            model = copy.deepcopy(model)
            model.reset_states(reset_model_parameters=True)
            trainer.fit(model, train_loader)

        val_loader = al_datamodule.custom_dataloader(val_idx, train=False)
        val_outputs = model.get_model_outputs(
            val_loader, output_types=['logits', 'labels'], device=self.device)
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

        return best_temp

    def _normalize_features(self, features):
        # I assume the paper uses L2 normalization, hence the following line:
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        # However, if you want to normalize by the maximum norm, you can use:
        # norms = torch.linalg.norm(features, ord=2, dim=1, keepdim=True)
        # max_norm = torch.max(norms)
        # features /= max_norm
        return features

    def _compute_uncertainty_scores(self, logits):
        if self.uncertainty == 'least_confidence':
            probs = logits.softmax(dim=-1)
            uncertainty_scores = 1 - probs.max(dim=-1).values
        elif self.uncertainty == 'margin':
            probs = logits.softmax(dim=-1)
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            uncertainty_scores = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif self.uncertainty == 'entropy':
            uncertainty_scores = -torch.sum(probs * probs.log(), dim=-1)
        else:
            raise NotImplementedError(f"Uncertainty type '{self.uncertainty}' is not implemented.")
        return uncertainty_scores


class MaxHerding(Query):
    def __init__(self, kernel='rbf', rbf_lengthscale=1, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed)
        self.subset_size = subset_size
        self.device = device

        if kernel == 'rbf':
            # https://github.com/BorealisAI/uherding/blob/main/deep-al/pycls/al/uherding.py#L90
            gamma = 1.0 / (rbf_lengthscale**2)
            self.kernel_func = lambda x, y: pairwise_kernels(x, y, metric='rbf', gamma=gamma)
        else:
            raise NotImplementedError(f"Kernel '{kernel}' is not implemented.")

    @torch.no_grad()
    def query(self, *, model, al_datamodule: ActiveLearningDataModule, acq_size):
        u_loader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        l_loader, _ = al_datamodule.labeled_dataloader()

        u_outputs = model.get_model_outputs(u_loader, output_types=['features'], device=self.device)
        l_outputs = model.get_model_outputs(l_loader, output_types=['features'], device=self.device)

        u_features = u_outputs['features']
        l_features = l_outputs.get('features', torch.empty(0, device=self.device))

        all_features = torch.cat((u_features, l_features), dim=0)
        all_features = self._normalize_features(all_features)
        u_features = all_features[:len(u_features)].cpu().numpy()
        l_features = all_features[len(u_features):].cpu().numpy()

        if len(l_features) != 0:
            K_ = self.kernel_func(u_features, l_features)
            max_kernels = K_.max(axis=1, keepdims=True)
        else:
            max_kernels = np.zeros((len(u_features), 1))
        K = self.kernel_func(u_features, u_features)

        selected = []
        for _ in range(acq_size):
            idx = np.argmax(np.mean(np.maximum(K - max_kernels, 0), axis=1))
            max_kernels = np.maximum(max_kernels, self.kernel_func(u_features, u_features[np.newaxis, idx]))
            selected.append(idx)

        return [unlabeled_indices[idx] for idx in selected]

    def _normalize_features(self, features):
        # I assume the paper uses L2 normalization, hence the following line:
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        # However, if you want to normalize by the maximum norm, you can use:
        # norms = torch.linalg.norm(features, ord=2, dim=1, keepdim=True)
        # max_norm = torch.max(norms)
        # features /= max_norm
        return features
