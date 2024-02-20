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
        probas = torch.softmax(logits, dim=-1) # Probas are required to be in the shape (N Samples, N Ensemble Members, N Classes)
        p_Yh_ThetaXp, p_Yh, p_Y_ThetaX = self.calculate_probabilities(probas=probas)
        indices = self.get_utilities(probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        return actual_indices
    

    def calculate_probabilities(self, probas: Tensor):
        ## Pr(y|theta,x)
        p_Y_ThetaX = probas
        p_Theta_L = 1 / p_Y_ThetaX.shape[1]

        ## Generate random number of x'
        xp_indices = random_generator_for_est_pool(p_Y_ThetaX.shape[0], self.estimation_pool_size)
        p_Yh_ThetaXp = p_Y_ThetaX[xp_indices, :, :]

        ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
        p_Y_ThetaX = p_Theta_L * p_Y_ThetaX
        p_Y_ThetaX = torch.transpose(p_Y_ThetaX, 1, 2)  ## transpose by dimension E and Y

        sum_p_Y_ThetaX = torch.sum(p_Y_ThetaX, dim=-1).unsqueeze(dim=-1)
        p_Theta_LXY = p_Y_ThetaX / sum_p_Y_ThetaX

        ## Calculate pr(y_hat)
        p_Theta_LXY = p_Theta_LXY.unsqueeze(dim=1)
        p_Yh = torch.matmul(p_Theta_LXY, p_Yh_ThetaXp)

        ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
        p_Yh_ThetaXp = p_Yh_ThetaXp.unsqueeze(dim = 0).unsqueeze(dim = 0)
        p_Yh_ThetaXp = p_Yh_ThetaXp.repeat(p_Yh.shape[0], p_Yh.shape[2], 1, 1, 1)

        p_Yh = p_Yh.unsqueeze(dim = 0)
        p_Yh = p_Yh.repeat(p_Yh_ThetaXp.shape[3],1,1,1,1)
        p_Yh = p_Yh.transpose(0,3).transpose(0,1)

        return p_Yh_ThetaXp, p_Yh, p_Y_ThetaX

    @abstractmethod
    def get_utilities(self, probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size):
        pass



class CoreLogTopKSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size):
        core_log = torch.mul(p_Yh_ThetaXp, torch.log(torch.div(p_Yh_ThetaXp, p_Yh)))
        core_log = torch.sum(torch.sum(core_log.sum(dim=-1), dim=-1),dim=-1)

        ## Calculate RR
        p_Y_LX = torch.sum(p_Y_ThetaX, dim=-1)
        rr = torch.sum(torch.mul(p_Y_LX, core_log), dim=-1) / p_Yh_ThetaXp.shape[2]

        indices = rr.topk(acq_size).indices.numpy()
        return indices



class CoreLogBatchSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size):
        core_log = torch.mul(p_Yh_ThetaXp, torch.log(torch.div(p_Yh_ThetaXp, p_Yh)))
        core_log = torch.sum(core_log.sum(dim=-1), dim=-1)
        core_log = torch.transpose(core_log, 1, 2)
        core_log = torch.transpose(core_log, 0, 1)

        ## Calculate RR
        p_Y_LX = torch.sum(p_Y_ThetaX, dim=-1)
        rr = p_Y_LX.unsqueeze(0) * core_log
        rr = torch.sum(rr, dim=-1)
        rr = torch.transpose(rr, 0, 1)

        indices = clustering(rr, probas, self.T, acq_size)
        return indices



class CoreMSETopKSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size):
        core_mse = (p_Yh_ThetaXp - p_Yh).pow(2)
        core_mse = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1), dim=-1)

        ## Calculate RR
        p_Y_LX = torch.sum(p_Y_ThetaX, dim=-1)
        rr = torch.sum(torch.mul(p_Y_LX, core_mse), dim=-1) / p_Yh_ThetaXp.shape[2]
        indices = rr.topk(acq_size).indices.numpy()
        return indices



class CoreMSEBatchSampling(BayesianEstimateSampling):
    def get_utilities(self, probas, p_Yh_ThetaXp, p_Yh, p_Y_ThetaX, acq_size):
        core_mse = (p_Yh_ThetaXp - p_Yh).pow(2)
        core_mse = torch.sum(core_mse.sum(dim=-1), dim=-1)
        core_mse = torch.transpose(core_mse, 1, 2)
        core_mse = torch.transpose(core_mse, 0, 1)

        ## Calculate RR
        p_Y_LX = torch.sum(p_Y_ThetaX, dim=-1)

        rr = p_Y_LX.unsqueeze(0) * core_mse
        rr = torch.sum(rr, dim=-1)
        rr = torch.transpose(rr, 0, 1)

        indices = clustering(rr, probas, self.T, acq_size)
        return indices