# Implementation of https://arxiv.org/abs/2205.11320. It is currently not working correctly.
# Code partially from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/prob_cover.py

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.utils.base import BaseModule


def calculate_purity(data, labels, delta) -> float:
    graph = construct_graph(data, delta)

    x = graph.x.to_numpy()
    y = graph.y.to_numpy()

    num_balls = labels.shape[0]  # There is a ball around every data point
    ball_indices = (x == y).nonzero()[0]  # Each balls starts here (With distance to itself)

    # Get labels of each elements in x and y
    x_labels = labels[x]
    y_labels = labels[y]

    label_equality = x_labels == y_labels

    # TODO (ynagel) This can probably be sped up (takes ~0.25 sec)
    # Calculates how many pure balls there are
    pure_balls = 0
    for i in range(len(ball_indices)):
        index = ball_indices[i]
        next_index = ball_indices[i + 1] if i + 1 < len(ball_indices) else len(label_equality)
        series = label_equality[index:next_index]
        false_count = np.count_nonzero(~series)

        if false_count == 0:
            pure_balls += 1

    return pure_balls / num_balls


def estimate_delta(data, num_classes, alpha=0.95):
    kmeans = KMeans(n_clusters=num_classes).fit(data)  # TODO (ynagel) Maybe consider MiniBatchKMeans
    labels = kmeans.labels_

    # TODO (ynagel) These values have to be computed and explained somehow
    deltas = np.linspace(1.0, 20.0, 1000)

    low = 0
    high = len(deltas) - 1
    delta = 1.0
    while low <= high:
        mid = (low + high) // 2
        purity = calculate_purity(data, labels, deltas[mid])

        if purity < alpha:
            high = mid - 1
        else:
            delta = deltas[mid]  # Current best delta
            low = mid + 1

    return delta


# TODO (ynagel) This has a rounding problem when calculating the distance of a point to itself, sometimes resulting in a
# point not being in its own ball.
def construct_graph(features, delta, batch_size=500):
    """
    creates a directed graph where:
    x->y iff l2(x,y) < delta.

    represented by a list of edges (a sparse matrix).
    stored in a dataframe
    """
    xs, ys, ds = [], [], []
    # distance computations are done in GPU
    cuda_feats = torch.tensor(features).cuda()  # TODO (ynagel) Check for availability
    for i in range(len(features) // batch_size):
        # distance comparisons are done in batches to reduce memory consumption
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(cur_feats, cuda_feats)
        mask = dist < delta
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T
        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    return df


class ProbCover(Query):

    def __init__(self, subset_size=None, random_seed=None, delta=0.6):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.delta = delta

    @torch.no_grad()
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: ActiveLearningDataModule,
              acq_size: int,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        unlabeled_features = model.get_representations(unlabeled_dataloader)
        if len(labeled_indices) > 0:
            labeled_features = model.get_representations(labeled_dataloader)
        else:
            labeled_features = torch.Tensor([])

        features = torch.cat([labeled_features, unlabeled_features])
        indices = labeled_indices + unlabeled_indices

        graph = construct_graph(features, self.delta)

        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(graph.x, np.arange(len(labeled_indices)))
        covered_samples = graph.y[edge_from_seen].unique()
        cur_df = graph[(~np.isin(graph.y, covered_samples))]
        for i in range(acq_size):
            coverage = len(covered_samples) / len(indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(indices))
            # print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == acq_size, 'added a different number of samples'
        selected_indices = np.array(indices, dtype=int)[selected].tolist()

        return selected_indices
