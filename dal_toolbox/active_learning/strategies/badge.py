import numpy as np

from scipy import stats
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

from .query import Query


# kmeans ++ initialization
def init_centers(X, K):
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
        eps = 1e-9
        Ddist = (D2 ** 2) / np.sum(D2 ** 2) + eps
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


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
        grad_embedding = model.get_grad_embedding(
            unlabeled_dataloader,
            n_samples=len(unlabeled_indices),
            device=self.device
        )
        chosen = init_centers(grad_embedding.numpy(), acq_size)
        return [unlabeled_indices[idx] for idx in chosen]


# class BadgeSampling(Strategy):
#     def __init__(self, X, Y, idxs_lb, net, handler, args):
#         super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
#
#     def query(self, n):
#         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
#         gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
#         chosen = init_centers(gradEmbedding, n)
#         return idxs_unlabeled[chosen]
#
