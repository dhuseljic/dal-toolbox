import numpy as np

from scipy import stats
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import stable_cumsum, row_norms

from .query import Query




class Badge(Query):
    def __init__(self, subset_size=None, batch_size=128, device='cuda'):
        super().__init__()
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
        del kwargs
        if not hasattr(model, 'get_grad_embedding'):
            raise ValueError('The method `get_grad_embedding` is mandatory to use badge sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)

        unlabeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)
        grad_embedding = model.get_grad_embedding(unlabeled_dataloader, device=self.device)
        chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, np_rng=self.np_rng)
        return [unlabeled_indices[idx] for idx in chosen]


def kmeans_plusplus(X, n_clusters, np_rng):
    _, indices = _kmeans_plusplus(
        X,
        n_clusters=n_clusters,
        x_squared_norms=row_norms(X, squared=True),
        random_state=np_rng,
    )
    indices = np.unique(indices).tolist()
    return indices


def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state):
    # kmeans++ from sklearn see:
    # https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/cluster/_kmeans.py#L164
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials
    # This is what Arthur/Vassilvitskii tried, but did not report
    # specific results for other than mentioning in the conclusion
    # that it helped.
    n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        indices[c] = best_candidate

    return centers, indices


# kmeans ++ initialization
def init_centers(X, K):
    # Code from: https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
    # # kmeans++ from algo in paper
    # np.random.seed(0)
    # idx = np.random.choice(range(len(X)))
    # indices = [idx]
    # centers = [X[idx]]
    # dist_mat = []
    # for i in range(K-1):
    #     dist = np.sqrt(np.sum((X - centers[-1])**2, axis=-1))
    #     dist_mat.append(dist)
    #     dist_mat_min= np.min(dist_mat, axis=0)
    #     dist_mat_squared = dist_mat_min**2
    #     if np.all(dist_mat_squared == 0):
    #         raise ValueError('All distances to the centers are zero!')
    #     p = dist_mat_squared / np.sum(dist_mat_squared)
    #     idx = np.random.choice(range(len(X)), p=p.squeeze())
    #     indices.append(idx)
    #     centers.append(X[idx])


    # grad_norm = np.linalg.norm(X, ord=2, axis=1)
    # ind = np.argmax(grad_norm)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(np.sum(D2)), flush=True)
        # if sum(D2) == 0.0:
        #     pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / np.sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            raise ValueError('All distances to the centers are zero!')
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll