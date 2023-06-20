import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import normalize
from scipy.special import rel_entr
from .query import Query
import torch
from torch import nn

class CAL(Query):

    def __init__(self, subset_size=None, batch_size=128, device='cuda'):
        super().__init__()
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.device = device

    def query(self, model, dataset, unlabeled_indices, labeled_indices, acq_size, **kwargs):
        if not hasattr(model, 'get_representations_and_probas'):
            raise ValueError('The method `get_representations_and_probas` is mandatory to use cal sampling.')

        if self.subset_size:
            unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)
        
        if "collator" in list(kwargs.keys()):
            unlabeled_dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size*2, 
                collate_fn=kwargs['collator'],
                sampler=unlabeled_indices)
            
            labeled_dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size*2, 
                collate_fn=kwargs['collator'],
                sampler=labeled_indices)
        else:
            unlabeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)
            labeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=labeled_indices)

        del kwargs

        # get embeddings and probas for pool and labeled data
        embs_pool, probas_pool = model.get_representations_and_probas(
            unlabeled_dataloader, 
            device=self.device)
        embs_labeled, probas_labeled = model.get_representations_and_probas(
            labeled_dataloader, 
            device=self.device)
        # get true targets for labeled set 
        y_labeled = torch.cat([i['labels'] for i in labeled_dataloader])

        # mean kl divergence score to 10 nearest neighbors and difference to labeled for every instance in unlabeled 
        kl_scores = self.cal(
            embs_pool.numpy(), 
            probas_pool.numpy(),
            embs_labeled.numpy(), 
            probas_labeled.numpy(),
            y_labeled.numpy(),
            n_neighbors=10,
        )

        # high score: high divergence in model predicted probas for candidate compared to neighbors in labeled set
        _, indices = torch.tensor(kl_scores).topk(acq_size)

        # reverse indices back to the true unlabeled indices 
        actual_indices = [unlabeled_indices[i] for i in indices]

        return actual_indices

    def cal(self, embs_pool, probas_pool, embs_labeled, probas_labeled, y_labeled, n_neighbors):
    # contrastive instances:
    # find data points that are similiar in the model feature space but the model outputs maximally different probas

        # 1) find embeddings that are similir in the model feature space:
            
        # get emb_labeled and emb_pool from [cls] (normalize them)
        embs_pool = normalize(embs_pool, axis=1)
        embs_labeled = normalize(embs_labeled, axis=1)

        # knn classifier fitted on embs_labeled data
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X=embs_labeled, y=y_labeled)        
        #dist = DistanceMetric.get_metric('euclidean')
        criterion = nn.KLDivLoss(reduction='none')

        # we want to query similar instances from the labeled neigborhood to each instance from the pool
        kl_scores = []
        #distances = []

        # go through every unlabeled sample in pool 
        for embs_candidate, probas_candidate in tqdm(zip(embs_pool, probas_pool)):
            # 2) find the 10 nearest neighbors of a candidate from pool
            # distances = distances from candidate embedding from pool to the k=10 nearest labeled neighbors (from KNN)
            # neigbhors_idx = indices of the nearest points in the embs_labeled to the embs_candidate
            _, neighbors_idx = neigh.kneighbors(X=[embs_candidate], return_distance=True)
            #distances.append(distances_[-1])

            # get the respective probas and preds from the nearest labeled instances --> n_neighbors x classes
            probas_labeled_neigh = [probas_labeled[n] for n in neighbors_idx][-1]
            #preds_labeled_neigh = [np.argmax(probas_labeled_neigh, axis=1)]
            # get the predictions of pool_candidate 
            #pred_candidate = [np.argmax(probas_candidate)]
            
            # input of kl divergence should be a distribution in the log space
            uda_softmax_temp = 1
            log_probas_candidate = torch.from_numpy(np.log(probas_candidate / uda_softmax_temp))

            # calculcate the KL Divergence between the candidate and the 10 nearest neighbor probas --> k=10 values
            kl = np.array([torch.sum(criterion(log_probas_candidate, n), dim=-1).numpy() for n in torch.from_numpy(probas_labeled_neigh)])
            # calculate a score for a candidate
            kl_candidate_score = kl.mean()
            # calculate a score for a
            kl_scores.append(kl_candidate_score)

        #distances = np.array([np.array(xi) for xi in distances])
        return kl_scores
