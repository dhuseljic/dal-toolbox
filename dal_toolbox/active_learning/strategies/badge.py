import numpy as np

from torch.utils.data import DataLoader

from .query import Query


class Badge(Query):
    def __init__(self, subset_size=None):
        super().__init__()
        self.subset_size = subset_size

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)

        grad_embedding = model.get_grad_representations(unlabeled_dataloader)
        chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, rng=self.rng)

        return [unlabeled_indices[idx] for idx in chosen]


def kmeans_plusplus(X, n_clusters, rng):
    # Start with highest grad norm since it is the "most uncertain"
    grad_norm = np.linalg.norm(X, ord=2, axis=1)
    idx = np.argmax(grad_norm)

    indices = [idx]
    centers = [X[idx]]
    dist_mat = []
    for _ in range(1, n_clusters):
        # Compute the distance of the last center to all samples
        dist = np.sqrt(np.sum((X - centers[-1])**2, axis=-1))
        dist_mat.append(dist)
        # Get the distance of each sample to its closest center
        min_dist = np.min(dist_mat, axis=0)
        min_dist_squared = min_dist**2
        if np.all(min_dist_squared == 0):
            raise ValueError('All distances to the centers are zero!')
        # sample idx with probability proportional to the squared distance
        p = min_dist_squared / np.sum(min_dist_squared)
        if np.any(p[indices] != 0):
            print('Already sampled centers have probability', p)
        idx = rng.choice(range(len(X)), p=p.squeeze())
        indices.append(idx)
        centers.append(X[idx])
    return indices

# kmeans ++ initialization
# def init_centers(X, K):
#     # Code from: https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
#     # first center is the one with highest gradient
#     ind = np.argmax([np.linalg.norm(s, 2) for s in X])
#     mu = [X[ind]]
#     indsAll = [ind]
#     centInds = [0.] * len(X)
#     cent = 0
#     print('#Samps\tTotal Distance')
#     while len(mu) < K:
#         if len(mu) == 1:
#             D2 = pairwise_distances(X, mu).ravel().astype(float)
#         else:
#             newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
#             for i in range(len(X)):
#                 if D2[i] > newD[i]:
#                     centInds[i] = cent
#                     D2[i] = newD[i]
#         print(str(len(mu)) + '\t' + str(np.sum(D2)), flush=True)
#         # if sum(D2) == 0.0:
#         #     pdb.set_trace()
#         D2 = D2.ravel().astype(float)
#         Ddist = (D2 ** 2) / np.sum(D2 ** 2)
#         customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
#         ind = customDist.rvs(size=1)[0]
#         while ind in indsAll:
#             raise ValueError('All distances to the centers are zero!')
#             ind = customDist.rvs(size=1)[0]
#         mu.append(X[ind])
#         indsAll.append(ind)
#         cent += 1
#     return indsAll
