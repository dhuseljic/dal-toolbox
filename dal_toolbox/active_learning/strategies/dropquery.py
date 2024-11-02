from .query import Query
from .utils import cluster_features, get_random_samples

import torch
import torch.nn.functional as F


class DropQuery(Query):
    def __init__(self, subset_size=None, num_iter=3, p_drop=0.75):
        super().__init__()
        self.subset_size = subset_size
        self.num_iter = num_iter
        self.p_drop = p_drop

    def _get_candidates(self, model, loader_unlabeled, y_star, acq_size):
        # Get self.num_iter many forward propagations of each unlabeled sample and extract the resutling label prediction
        labels = []
        model.model.set_dropout(p=self.p_drop)
        with torch.no_grad():
            for _ in range(self.num_iter):
                lab = []
                for batch in loader_unlabeled:
                    samples = batch[0]
                    logits = model.model.forward_dropout(samples)
                    lab.append(logits.softmax(-1).argmax(-1))
                lab = torch.cat(lab)
                labels.append(lab)
        labels = torch.stack(labels)

        # Then count the number of missmatches to the originally predicted label
        mismatch = (y_star != labels).sum(dim=0)

        # Countinously reduce the threshhold for missmatches to be selected as a candidate until threshhold is 1 or there are more then 25 times the query size of candidates 
        thresh = self.num_iter // 2
        while (mismatch > thresh).sum() < 25 * acq_size and thresh > 0:
            thresh = thresh - 1

        # Return indexlist of unlabeled samples that are sufficiently uncertain
        return torch.nonzero(mismatch > thresh).flatten()
    
    def get_emb_probs(self, dataloader, model):
        embeddings, probs = [], []
        with torch.no_grad():
            for batch in dataloader:
                samples = batch[0]
                log, emb = model(samples, return_features=True)
                embeddings.append(emb)
                probs.append(log.softmax(-1))

        embeddings = torch.cat(embeddings)
        probs = torch.cat(probs)

        return embeddings, probs

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        loader_unlabeled, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        num_unlabeled = len(unlabeled_indices)

        embeddings, probs = self.get_emb_probs(dataloader=loader_unlabeled, model=model)
        y_star = probs.argmax(dim=1)
        candidates = self._get_candidates(model, loader_unlabeled, y_star, acq_size)

        if len(candidates) < acq_size:
            delta = acq_size - len(candidates)
            random_samples = get_random_samples(candidates, delta, num_unlabeled)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)
        else:
            candidate_vectors = F.normalize(embeddings[candidates]).numpy()
            selected, _ = cluster_features(candidate_vectors, acq_size)

        selected_candidates = candidates[selected].tolist()
        query_indices = [unlabeled_indices[i] for i in selected_candidates]
        return query_indices