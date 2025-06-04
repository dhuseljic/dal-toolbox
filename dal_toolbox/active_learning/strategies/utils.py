from sklearn.cluster import KMeans
import torch
import numpy as np
import random


def cluster_features(features, acq_size, weights=None):
    # Fit kmeans on the input features
    kmeans = KMeans(n_clusters=acq_size)
    kmeans.fit(X=features, sample_weight=weights)

    # Calculate the square_distances to the closest clusters
    all_sq_dist = kmeans.transform(X=features)
    all_sq_dist = torch.tensor(all_sq_dist)
    sq_dist, cluster_idx = torch.min(all_sq_dist, dim=-1)

    selected = []
    for i in range(acq_size):
        idx = torch.nonzero(cluster_idx == i).ravel()
        if idx.shape[0] > 0:
            min_idx = sq_dist[idx].argmin()  # point closest to the centroid
            selected.append(idx[min_idx].item())  # add that id to the selected set

    # Fill up selected with random samples up to acq_size
    diff = acq_size - len(selected)
    selected = selected + random.sample([i for i in range(features.shape[0]) if i not in selected], k=diff)
    return selected, cluster_idx


def get_random_samples(candidates, acq_size, num_unlabeled):
    unlabeled_pool = torch.ones(num_unlabeled, dtype=torch.bool)
    unlabeled_pool[candidates] = False  # all candidates will be labeled
    remaining = torch.nonzero(unlabeled_pool).flatten()
    idx = np.random.choice(len(remaining), acq_size, replace=False)
    return remaining[idx]