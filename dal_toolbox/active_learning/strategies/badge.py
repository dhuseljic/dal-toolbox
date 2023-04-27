import numpy as np

from torch.utils.data import DataLoader

from .query import Query


class Badge(Query):
    def __init__(self, subset_size=None, batch_size=128, device='cuda'):
        super().__init__()
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
        if not hasattr(model, 'get_grad_embedding'):
            raise ValueError('The method `get_grad_embedding` is mandatory to use badge sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.choice(unlabeled_indices, size=self.subset_size, replace=False)

        unlabeled_dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                          sampler=unlabeled_indices, collate_fn=kwargs.get("collator"))

        grad_embedding = model.get_grad_embedding(unlabeled_dataloader, device=self.device)
        chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, np_rng=self.np_rng)
        return [unlabeled_indices[idx] for idx in chosen]


def kmeans_plusplus(X, n_clusters, np_rng):
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
        idx = np_rng.choice(range(len(X)), p=p.squeeze())
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
