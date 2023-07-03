import numpy as np
import pandas as pd
import torch

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query
from dal_toolbox.models.utils.base import BaseModule


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

        graph = self.construct_graph(features)

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

    def construct_graph(self, features, batch_size=500):
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
            mask = dist < self.delta
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
