import torch
import random
import numpy as np

from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from scipy.spatial.distance import cdist

from torch import Tensor
from abc import ABC, abstractmethod
from .query import Query
from ..data import ActiveLearningDataModule
from ...models.utils.base import BaseModule


def random_generator_for_est_pool(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)


## cluster methods
def clustering(rr_X_Xp, probs_B_K_C, T, batch_size):
    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, round(probs_B_K_C.shape[0] * T))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()
    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]

    rr_X_Xp = normalize(rr_X_Xp)
    # rr_X_Xp = convert_embedding_by_tsne(rr_X_Xp)

    rr = kmeans(rr_X_Xp, batch_size)
    rr = [rr_topk_X_indices[x] for x in rr]

    return rr


## sub fuction for kmeans ++
def closest_center_dist(rr, centers):
    dist = torch.cdist(rr, rr[centers])
    cd = dist.min(axis=1).values
    return cd


## kmeans
def kmeans(rr, k):
    kmeans = KMeans(n_clusters=k).fit(rr)
    centers = kmeans.cluster_centers_
    # find the nearest point to centers
    centroids = cdist(centers, rr).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(rr)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids


class BayesianEstimateSampling(Query, ABC):
    def __init__(self, estimation_pool_size, T_fraction, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.estimation_pool_size, self.T = estimation_pool_size, T_fraction

    @torch.no_grad()
    def query(
        self,
        *,
        model: BaseModule,
        al_datamodule: ActiveLearningDataModule,
        acq_size: int,
        return_utilities: bool = False,
        # forward_kwargs: dict = None, TODO
        **kwargs
    ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader) # , **forward_kwargs)
        if logits.ndim != 3:
            raise ValueError(f"Input logits tensor must be 3-dimensional, got shape {logits.shape}")
        probas = torch.softmax(logits, dim=-1)
        pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh = self.calculate_stuff(probas=probas)
        indices = self.get_utilities(probas, pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh, acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices
    

    def calculate_stuff(self, probas: Tensor):
        ## Pr(y|theta,x)
        pr_YThetaX_X_E_Y = probas
        pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

        ## Generate random number of x'
        xp_indices = random_generator_for_est_pool(pr_YThetaX_X_E_Y.shape[0], self.estimation_pool_size)
        pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

        ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
        pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
        pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

        sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
        pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

        ## Calculate pr(y_hat)
        pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
        pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

        ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
        pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
        pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

        pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
        pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
        pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

        return pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh

    @abstractmethod
    def get_utilities(self, pr_Yhat_X_Xp_Y_Yh, pr_YhThetaXp_Xp_E_Yh, pr_YThetaX_X_Y_E, acq_size):
        pass


class CoreLogTopKSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh, acq_size):
        core_log = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.log(torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh)))
        core_log_X_Y = torch.sum(torch.sum(core_log.sum(dim=-1), dim=-1),dim=-1)

        ## Calculate RR
        pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
        rr = torch.sum(torch.mul(pr_YLX_X_Y, core_log_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]

        indices = rr.topk(acq_size).indices.numpy()
        return indices
    
class CoreLogBatchSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh, acq_size):
        core_log = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.log(torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh)))
        core_log_X_Y_Xp = torch.sum(core_log.sum(dim=-1), dim=-1)
        core_log_X_Xp_Y = torch.transpose(core_log_X_Y_Xp, 1, 2)
        core_log_Xp_X_Y = torch.transpose(core_log_X_Xp_Y, 0, 1)

        ## Calculate RR
        pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
        rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_log_Xp_X_Y
        rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
        rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

        indices = clustering(rr_X_Xp, probas, self.T, acq_size)
        return indices

class CoreMSETopKSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh, acq_size):
        core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
        core_mse_X_Y = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1), dim=-1)

        ## Calculate RR
        pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
        rr = torch.sum(torch.mul(pr_YLX_X_Y, core_mse_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]
        indices = rr.topk(acq_size).indices.numpy()
        return indices

class CoreMSEBatchSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh, pr_YThetaX_X_Y_E, pr_YhThetaXp_Xp_E_Yh, acq_size):
        core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
        core_mse_X_Y_Xp = torch.sum(core_mse.sum(dim=-1), dim=-1)
        core_mse_X_Xp_Y = torch.transpose(core_mse_X_Y_Xp, 1, 2)
        core_mse_Xp_X_Y = torch.transpose(core_mse_X_Xp_Y, 0, 1)

        ## Calculate RR
        pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)

        rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_mse_Xp_X_Y
        rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
        rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

        indices = clustering(rr_X_Xp, probas, self.T, acq_size)
        return indices
