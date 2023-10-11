# TODO (ynagel) There is currently a lot of redundancy in this code

import numpy as np
import pandas as pd
import torch

from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.active_learning.strategies.typiclust import kmeans
from dal_toolbox.active_learning.strategies.xpal import xpal_gain


class XPALClust(Query):
    """XPALClust
    The expected probabilistic active learning (xPAL) strategy combined with the clustering strategy from TypiClust.
    Parameters
    ----------
    n_classes: int
        Number of classes.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    subset_size: int
        How much of the unlabeled dataset is taken into account.
    random_seed: numeric | np.random.RandomState
        Random seed for annotator selection.
    Attributes
    ----------
    n_classes_: int
        Number of classes.
    S_: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    """
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, num_classes, S, alpha_c, alpha_x, subset_size=None, random_seed=None):
        super().__init__(random_seed)

        self.num_classes = num_classes
        self.S = S
        self.alpha_c_ = alpha_c
        self.alpha_x_ = alpha_x
        self.subset_size = subset_size

    def query(self, *, model, al_datamodule, acq_size, return_gains=False, **kwargs):
        """Compute score for each unlabeled sample. Score is to be maximized.
        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled sample.
        """
        # compute frequency estimates for evaluation set (K_x) and candidate set (K_c)
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_loader, labeled_indices = al_datamodule.labeled_dataloader()

        if self.subset_size is None:
            S_ = self.S

            mapped_labeled_indices = labeled_indices
            mapped_unlabeled_indices = unlabeled_indices
        else:
            # Deletes entries from S, that are not part of the labeled/unlabeled subset
            existing_indices = np.concatenate([unlabeled_indices, labeled_indices])
            indices_to_remove = np.arange(self.S.shape[0])
            mask = np.equal.outer(indices_to_remove, existing_indices)
            indices_to_remove = indices_to_remove[~np.logical_or.reduce(mask, axis=1)]

            S_ = np.delete(self.S, indices_to_remove, 0)
            S_ = np.delete(S_, indices_to_remove, 1)

            # Remapping indices
            mapping = np.argsort(np.argsort(existing_indices))
            mapped_labeled_indices = mapping[len(unlabeled_indices):]
            mapped_unlabeled_indices = mapping[:len(unlabeled_indices)]

        y_labeled = []

        for batch in labeled_loader:
            y_labeled.append(batch[1])
        if len(y_labeled) > 0:
            y_labeled = torch.cat(y_labeled).tolist()

        Z = np.eye(self.num_classes)[y_labeled]
        K_x = S_[:, mapped_labeled_indices] @ Z
        K_c = K_x[mapped_unlabeled_indices]

        # calculate loss reduction for each unlabeled sample
        gains = xpal_gain(K_c=K_c, K_x=K_x, S=S_[mapped_unlabeled_indices], alpha_c=self.alpha_c_,
                          alpha_x=self.alpha_x_)

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_features = model.get_representations(unlabeled_dataloader)
        if len(labeled_indices) > 0:
            labeled_features = model.get_representations(labeled_loader)
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
        top_gains = []

        for i in range(acq_size):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0] - len(labeled_indices)
            rel_gains = gains[indices]
            idx = indices[rel_gains.argmax()]
            selected.append(idx)
            top_gains.append(rel_gains.max())
            labels[idx] = -1

        selected = np.array(selected)
        actual_indices = [unlabeled_indices[i] for i in selected]

        if return_gains:
            return actual_indices, top_gains
        return actual_indices
