import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.utils.base import BaseModule

try:
    import faiss
except ImportError:
    faiss = None


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    # features = features.astype(np.float32)
    features = features.numpy().astype(np.float32)

    if faiss is not None:
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(features)  # add vectors to the index
        distances, indices = cpu_index.search(features, num_neighbors + 1)
        # if os.name == 'nt':  # faiss_gpu is not implemented on windows
        #     cpu_index.add(features)  # add vectors to the index
        #     distances, indices = cpu_index.search(features, num_neighbors + 1)
        # else:
        #     gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        #     gpu_index.add(features)  # add vectors to the index
        #     distances, indices = gpu_index.search(features, num_neighbors + 1)
    else:
        raise NotImplementedError("TypiClust is currently not implemented without faiss. "
                                  "(https://github.com/facebookresearch/faiss)")

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
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init='auto')
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClust(Query):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, subset_size=None, random_seed=None, precomputed=True):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.precomputed = precomputed

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

        if self.precomputed:
            unlabeled_features = torch.cat([batch[0] for batch in unlabeled_dataloader])
            if len(labeled_indices) > 0:
                labeled_features = torch.cat([batch[0] for batch in labeled_dataloader])
            else:
                labeled_features = torch.Tensor([])
        else:
            unlabeled_features = model.get_representations(unlabeled_dataloader)
            if len(labeled_indices) > 0:
                labeled_features = model.get_representations(labeled_dataloader)
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
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        actual_indices = [unlabeled_indices[i - len(labeled_indices)] for i in selected]
        return actual_indices
