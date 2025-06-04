from .query import Query
from .utils import cluster_features, get_random_samples

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader


class DropQuery(Query):
    def __init__(self, subset_size=None, num_iter=3, p_drop=0.75, device='cpu'):
        super().__init__()
        self.subset_size = subset_size
        self.num_iter = num_iter
        self.dropout = nn.Dropout(p=p_drop)
        self.device = device

    def _get_candidates(self, model, features, y_star, acq_size):
        model = model.to(self.device)
        feature_loader = DataLoader(features, batch_size=64, shuffle=False)

        # Get self.num_iter forward propagations of each unlabeled sample and extract the prediction
        labels = []
        for _ in range(self.num_iter):
            predictions = []
            for feature in feature_loader:
                feature = feature.to(self.device)
                feature_drop = self.dropout(feature)
                pred = model.model.forward_head(feature_drop).softmax(-1).argmax(-1).cpu()
                predictions.append(pred)
            predictions = torch.cat(predictions)
            labels.append(predictions)
        labels = torch.stack(labels)

        # Then count the number of missmatches to the originally predicted label
        mismatch = (y_star != labels).sum(dim=0)

        # Countinously reduce the threshhold for missmatches to be selected as a candidate until threshhold is 1 or there are more then 25 times the query size of candidates
        thresh = self.num_iter // 2
        while (mismatch > thresh).sum() < 25 * acq_size and thresh > 0:
            thresh = thresh - 1

        # Return indexlist of unlabeled samples that are sufficiently uncertain
        return torch.nonzero(mismatch > thresh).flatten()

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        if not hasattr(model.model, 'forward_head'):
            raise RuntimeError("""The attribute 'forward_head' is required to use DropQuery, but it was not 
                               found in model. Please ensure it is defined.""")

        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        num_unlabeled = len(unlabeled_indices)

        outputs = model.get_model_outputs(unlabeled_dataloader, ['features', 'logits'], device=self.device)
        features = outputs['features']
        logits = outputs['logits']
        y_star = logits.softmax(-1).argmax(-1)
        candidates = self._get_candidates(model, features, y_star, acq_size)

        if len(candidates) < acq_size:
            delta = acq_size - len(candidates)
            random_samples = get_random_samples(candidates, delta, num_unlabeled)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)
        else:
            candidate_vectors = F.normalize(features[candidates]).numpy()
            selected, _ = cluster_features(candidate_vectors, acq_size)

        selected_candidates = candidates[selected].tolist()
        query_indices = [unlabeled_indices[i] for i in selected_candidates]
        return query_indices
