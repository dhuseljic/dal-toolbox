# Implementation of https://arxiv.org/abs/2202.02794.
# Code partially from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.utils.base import BaseModule


def get_nn(features, num_neighbors):
    features = features.numpy().astype(np.float32)
    nn_calculator = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='sqeuclidean', n_jobs=-1).fit(features)
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
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init='auto')
    km.fit_predict(features)
    # if num_clusters <= 50:
    #     km = KMeans(n_clusters=num_clusters, n_init='auto')
    #     km.fit_predict(features)
    # else:
    #     km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, reassignment_ratio=0.0)
    #     km.fit_predict(features)
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

        unlabeled_features = model.get_representations(unlabeled_dataloader, device=self.device)
        if len(labeled_indices) > 0:
            labeled_features = model.get_representations(labeled_dataloader, device=self.device)
        else:
            labeled_features = torch.Tensor([])

        features = torch.cat((labeled_features, unlabeled_features))
        clusters = kmeans(features, num_clusters=num_clusters)

        labels = clusters.copy()
        existing_indices = np.arange(len(labeled_indices))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame(
            {'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
             'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        for i in range(acq_size):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            
            if len(rel_feats) == 0: # in case cluster is empty skip the cluster, random sample rest
                continue

            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        actual_indices = [unlabeled_indices[i - len(labeled_indices)] for i in selected]

        if len(selected) != acq_size:
            # Randomly sample the rest
            filtered_indices = set(unlabeled_indices).difference(actual_indices)
            idx = self.rng.choice(list(filtered_indices), size=acq_size - len(selected), replace=False)
            actual_indices  = actual_indices + idx.tolist()
            
        return actual_indices


        n_clusters = len(labeled_sample_indices) + batch_size
        cluster_algo_dict[self.n_cluster_param_name] = n_clusters
        cluster_obj = self.cluster_algo(**cluster_algo_dict)

        cluster_labels = cluster_obj.fit_predict(X)

        # determine number of samples per cluster and mask clusters with
        # labeled samples
        cluster_sizes = np.zeros(n_clusters)
        cluster_ids, cluster_ids_sizes = np.unique(
            cluster_labels, return_counts=True
        )
        cluster_sizes[cluster_ids] = cluster_ids_sizes
        covered_cluster = np.unique(
            [cluster_labels[i] for i in labeled_sample_indices]
        )
        if len(covered_cluster) > 0:
            cluster_sizes[covered_cluster] = 0

        utilities = np.full(shape=(batch_size, X.shape[0]), fill_value=np.nan)
        query_indices = []
        for i in range(batch_size):
            if cluster_sizes.max() == 0:
                typicality = np.ones(len(X))
            else:
                cluster_id = rand_argmax(
                    cluster_sizes, random_state=self.random_state_
                )
                is_cluster = cluster_labels == cluster_id
                uncovered_samples_mapping = np.where(is_cluster)[0]
                typicality = _typicality(X, uncovered_samples_mapping, self.k)
            utilities[i, mapping] = typicality[mapping]
            utilities[i, query_indices] = np.nan
            idx = rand_argmax(
                typicality[mapping], random_state=self.random_state_
            )
            idx = mapping[idx[0]]

            query_indices = np.append(query_indices, [idx]).astype(int)
            cluster_sizes[cluster_id] = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices


def _typicality(X, uncovered_samples_mapping, k, eps=1e-7):
    """
    Calculation the typicality of samples `X` in uncovered clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    uncovered_samples_mapping : np.ndarray of shape (n_candidates,),
    default=None
       Index array that maps `candidates` to `X_for_cluster`.
    k : int
        k for computation of k nearst neighbors.
    eps : float > 0, default=1e-7
        Minimum distance sum to compute typicality.

    Returns
    -------
    typicality : numpy.ndarray of shape (n_X)
        The typicality of all uncovered samples in X
    """
    typicality = np.full(shape=X.shape[0], fill_value=-np.inf)
    if len(uncovered_samples_mapping) == 1:
        typicality[uncovered_samples_mapping] = 1
        return typicality
    k = np.min((len(uncovered_samples_mapping) - 1, k))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X[uncovered_samples_mapping])
    dist_matrix_sort_inc, _ = nn.kneighbors(
        X[uncovered_samples_mapping], n_neighbors=k + 1, return_distance=True
    )
    knn = np.sum(dist_matrix_sort_inc, axis=1) + eps
    typi = ((1 / k) * knn) ** (-1)
    typicality[uncovered_samples_mapping] = typi
    return typicality