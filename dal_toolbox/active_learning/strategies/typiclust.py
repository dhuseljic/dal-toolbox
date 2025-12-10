# Implementation of https://arxiv.org/abs/2202.02794.
# Code partially from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.base import BaseModule


def get_nn(features, num_neighbors):
    features = features.numpy().astype(np.float32)
    nn_calculator = NearestNeighbors(n_neighbors=num_neighbors + 1,
                                     metric='sqeuclidean', n_jobs=-1).fit(features)
    distances, indices = nn_calculator.kneighbors(features)

    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    if len(features) > 1: 
        mean_distance = get_mean_nn_dist(features, num_neighbors)
        typicality = 1 / (mean_distance + 1e-5)
    else: 
        typicality = np.array([np.inf])
    return typicality


def kmeans(features, num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init='auto')
    km.fit_predict(features)
    return km.labels_


class TypiClust(Query):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, subset_size=None, random_seed=None, device='cpu'):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.device = device

    @torch.no_grad()
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: ActiveLearningDataModule,
              acq_size: int,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, ['features'], device=self.device)
        unlabeled_features = unlabeled_outputs['features']
        if len(labeled_indices) > 0:
            labeled_outputs = model.get_model_outputs(labeled_dataloader, ['features'], device=self.device)
            labeled_features = labeled_outputs['features']
        else:
            labeled_features = torch.Tensor([])
        features = torch.cat((labeled_features, unlabeled_features))

        # See https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/pool/_typi_clust.py
        clusters = kmeans(features, num_clusters=num_clusters)
        cluster_sizes = np.zeros(num_clusters)
        cluster_ids, cluster_id_sizes = np.unique(clusters, return_counts=True)
        cluster_sizes[cluster_ids] = cluster_id_sizes
        covered_clusters = np.unique(clusters[:len(labeled_indices)])
        if len(covered_clusters) > 0:
            cluster_sizes[covered_clusters] = 0

        query_indices = []
        for i in range(acq_size):
            if cluster_sizes.max() == 0:
                indices_ = np.arange(len(unlabeled_features))
                indices_ = np.setdiff1d(indices_, query_indices)
                idx = self.rng.choice(indices_)
                query_indices.append(idx)
            else:
                cluster_id = cluster_sizes.argmax()
                cluster_indices = (clusters == cluster_id).nonzero()[0]
                cluster_features = features[cluster_indices]
                typicality = calculate_typicality(cluster_features, min(self.K_NN, len(cluster_indices) // 2))

                idx = typicality.argmax()
                idx = cluster_indices[idx]
                query_indices.append(idx-len(labeled_features))
                cluster_sizes[cluster_id] = 0

        actual_indices = [unlabeled_indices[idx] for idx in query_indices]
        return actual_indices


class InverseTypiClust(TypiClust):
    @torch.no_grad()
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: ActiveLearningDataModule,
              acq_size: int,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_outputs = model.get_model_outputs(unlabeled_dataloader, ['features'], device=self.device)
        unlabeled_features = unlabeled_outputs['features']
        if len(labeled_indices) > 0:
            labeled_outputs = model.get_model_outputs(labeled_dataloader, ['features'], device=self.device)
            labeled_features = labeled_outputs['features']
        else:
            labeled_features = torch.Tensor([])
        features = torch.cat((labeled_features, unlabeled_features))

        # See https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/pool/_typi_clust.py
        clusters = kmeans(features, num_clusters=num_clusters)
        cluster_sizes = np.zeros(num_clusters)
        cluster_ids, cluster_id_sizes = np.unique(clusters, return_counts=True)
        cluster_sizes[cluster_ids] = cluster_id_sizes
        covered_clusters = np.unique(clusters[:len(labeled_indices)])
        if len(covered_clusters) > 0:
            cluster_sizes[covered_clusters] = 0

        query_indices = []
        for i in range(acq_size):
            if cluster_sizes.min() == np.inf:
                indices_ = np.arange(len(unlabeled_features))
                indices_ = np.setdiff1d(indices_, query_indices)
                idx = self.rng.choice(indices_)
                query_indices.append(idx)
            else:
                cluster_id = cluster_sizes.argmin()
                cluster_indices = (clusters == cluster_id).nonzero()[0]
                cluster_features = features[cluster_indices]
                typicality = calculate_typicality(cluster_features, min(self.K_NN, len(cluster_indices) // 2))

                idx = typicality.argmin()
                idx = cluster_indices[idx]
                query_indices.append(idx-len(labeled_features))
                cluster_sizes[cluster_id] = np.inf

        actual_indices = [unlabeled_indices[idx] for idx in query_indices]
        return actual_indices