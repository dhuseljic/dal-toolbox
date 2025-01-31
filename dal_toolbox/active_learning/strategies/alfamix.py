from .query import Query
from .utils import cluster_features, get_random_samples

import numpy as np
import torch
import torch.nn.functional as F


class AlfaMix(Query):
    def __init__(self, embed_dim, subset_size=None, device='cpu'):
        super().__init__()
        self.subset_size = subset_size
        D = embed_dim
        self.eps = 0.2 / np.sqrt(D)
        self.device=device

    def _get_anchors(self, features: torch.Tensor, labels):
        seen_classes = len(np.unique(labels))
        anchors = []

        for i in range(seen_classes):
            anchor = features[labels == i].mean(dim=0)
            anchors.append(anchor)

        return torch.stack(anchors)

    def _get_candidates(self, z_u, z_star, z_grad, y_star, model):
        candidates = torch.zeros_like(y_star, dtype=torch.bool)
        grad_norm = torch.linalg.vector_norm(z_grad, dim=-1, keepdim=True)

        for z_s in z_star:
            z_diff = z_s - z_u
            diff_norm = torch.linalg.vector_norm(z_diff, dim=1, keepdim=True)
            alpha = self.eps * (diff_norm * z_grad) / (grad_norm * z_diff)
            z_lerp = alpha * z_s + (1 - alpha) * z_u

            probs = model.model.linear(z_lerp).softmax(dim=1)
            y_pred = torch.argmax(probs, dim=1)
            mismatch_idx = torch.nonzero(y_pred != y_star).flatten()
            candidates[mismatch_idx] = True

        return torch.nonzero(candidates).flatten()

    def query(self, model, al_datamodule, acq_size):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_dataloader, _ = al_datamodule.labeled_dataloader()

        # Retrieve gradient-embeddings, embeddings and pseudo-labels for the unlabeled data
        grads, z_u, y_star = model.get_alfa_grad_representations(unlabeled_dataloader, device=self.device)

        # Get the anchors, i.e., the centroids in the embedding space, of each observed class based on the labeled pool
        z_l, y_l = model.get_representations(dataloader=labeled_dataloader, return_labels=True, device=self.device)
        z_star = self._get_anchors(z_l, y_l).to(self.device)
        
        candidates = self._get_candidates(z_u.to(self.device), z_star.to(self.device), grads.to(self.device), y_star.to(self.device), model=model.to(self.device))

        # When there are not enough candidates, fill them up with random selection.
        # Otherwise cluster them and pick the closest to the cluster centroids.
        if len(candidates) < acq_size:
            delta = acq_size - len(candidates)
            random_samples = get_random_samples(candidates, delta, num_unlabeled=len(unlabeled_indices))
            candidates = torch.cat([candidates.cpu(), random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)
        else:
            candidate_vectors = F.normalize(z_u[candidates.cpu()]).cpu().numpy()
            selected, _ = cluster_features(candidate_vectors, acq_size, weights=None)

        selected_candidates = candidates[selected].tolist()
        query_indices = [unlabeled_indices[i] for i in selected_candidates]
        return query_indices

# @torch.enable_grad()
# def get_alfa_grad_representations( dataloader, device):
#     self.to(device)

#     embeddings, gradients, pseudo_labels = [], [], []
#     for batch in dataloader:
#         features = batch[0].to(device).requires_grad_()
#         logits = self(features)
#         preds = logits.argmax(-1)
#         loss = F.cross_entropy(logits, preds, reduction="sum")
#         grads = torch.autograd.grad(loss, features)[0]

#         embeddings.append(features.cpu().detach())
#         gradients.append(grads.cpu())
#         pseudo_labels.append(preds.cpu())

#     # Concat all tensors and return
#     gradients = torch.cat(gradients)
#     embeddings = torch.cat(embeddings)
#     pseudo_labels = torch.cat(pseudo_labels)

#     return gradients, embeddings, pseudo_labels
