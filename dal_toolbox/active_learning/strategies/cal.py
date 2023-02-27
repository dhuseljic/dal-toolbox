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

        embeddings_unlabeled, probas_unlabeled = model.get_representations_and_probas(
            unlabeled_dataloader, 
            device=self.device)
        embeddings_labeled, probas_labeled = model.get_representations_and_probas(
            labeled_dataloader, 
            device=self.device)
        
        y_labeled = torch.cat([i['labels'] for i in labeled_dataloader])
        # score for every instance in unlabeled pool 
        kl_scores = self.cal(
            embeddings_unlabeled.numpy(), 
            probas_unlabeled.numpy(),
            embeddings_labeled.numpy(), 
            probas_labeled.numpy(),
            y_labeled.numpy(),
            n_neighbors=10,
            acq_size=acq_size

        )
        # score for every instance in unlabeled pool 
        #selected_inds = np.argpartition(kl_scores, -acq_size)[-acq_size:]
        _, indices = torch.tensor(kl_scores).topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]

        return actual_indices

    def cal(self, embeddings_unlabeled, probas_unlabeled, embeddings_labeled, probas_labeled, y_labeled, n_neighbors, acq_size):
        # centroids: unlabeled datapoints?
        embeddings_unlabeled = normalize(embeddings_unlabeled, axis=1)
        embeddings_labeled = normalize(embeddings_labeled, axis=1)

        # knn classifier fitted on labeled data
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X=embeddings_labeled, y=y_labeled)        
        dist = DistanceMetric.get_metric('euclidean')
        criterion = nn.KLDivLoss(reduction='none')

        kl_scores = []
        num_adv = 0
        distances = []
        pairs = []
        # go throiugh every unlabeled sample in pool 
        for _, candidate in enumerate(
                tqdm(zip(embeddings_unlabeled, probas_unlabeled), desc="Finding neighbours for every unlabeled data point")):
            # find indices of closesest "neighbours" in train set
            #unlab_representation, unlab_logit = candidate
            # distances = distances from candidate to each center (10?)
            # neighors_idx = indices of the nearest points in the popluation matrix (labeled data!)
            distances_, neighbors_idx = neigh.kneighbors(X=[candidate[0]], return_distance=True)
            distances.append(distances_[0])

            # labeled_neighbours_inds = np.array(labeled_inds)[neighbors_idx[0]]  # orig inds
            # labeled_neighbours_labels = train_dataset.tensors[3][neighbors_idx[0]]
            # calculate score

            # probas_neigh = probabilities of the nearest points in the population (labeled data) --> n_neighbors x classes
            probas_neigh = [probas_labeled[n] for n in neighbors_idx][-1]
            probas_candidate = candidate[1] # probas for prediction of current candidate (from unlabeled pool!)

            # logits_neigh = [train_logits[n] for n in neighbors_idx]
            # logits_candidate = candidate[1]
            #neigh_prob = F.softmax(train_logits[neighbors_idx], dim=-1)

            # predictions of labeled 
            preds_neigh = [np.argmax(probas_labeled[n], axis=1) for n in neighbors_idx]
            # predictions of candidate
            pred_candidate = [np.argmax(probas_candidate)]
            # number of predictions that exist in preds_neigh and pred_candidate
            num_diff_pred = len(list(set(preds_neigh[-1]).intersection(pred_candidate)))

            if num_diff_pred > 0: num_adv += 1
            uda_softmax_temp = 1
            candidate_log_prob = torch.from_numpy(np.log(candidate[1] / uda_softmax_temp))
            kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in torch.from_numpy(probas_neigh)])
            kl_scores.append(kl.mean())
        distances = np.array([np.array(xi) for xi in distances])

        #logger.info('Total Different predictions for similar inputs: {}'.format(num_adv))

        #selected_inds = np.argpartition(kl_scores, -acq_size)[-acq_size:]
        return kl_scores

    #neigh.fit(X=embeddings_labeled, y=y_labeled)
    

    # embeddings = normalize(embeddings, axis=1)
    # nn = NearestNeighbors(n_neighbors=n_neighbors)
    # nn.fit(embeddings)

    # num_batches = int(np.ceil(len_dataset / self.batch_size))
    # offset = 0

    
    # for batch_idx in np.array_split(np.arange(embeddings.shape[0]), num_batches,
    #                                 axis=0):

    #     nn_indices = nn.kneighbors(embeddings[batch_idx],
    #                                n_neighbors=self.k,
    #                                return_distance=False)

    #     kl_divs = np.apply_along_axis(lambda v: np.mean([
    #         rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
    #         for i in nn_indices[v - offset]]),
    #         0,
    #         batch_idx[None, :])

    #     scores.extend(kl_divs.tolist())
    #     offset += batch_idx.shape[0]

    # scores = np.array(scores)
    # indices = np.argpartition(-scores, n)[:n]

    # return indices

    


    #chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, np_rng=self.np_rng)
    #return [unlabeled_indices[idx] for idx in chosen]
    

