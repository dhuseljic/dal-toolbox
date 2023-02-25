# import numpy as np

# from torch.utils.data import DataLoader
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import cdist
# from .query import Query

# class CAL(Query):

#     def __init__(self, subset_size=None, batch_size=128, device='cuda'):
#         super().__init__()
#         self.subset_size = subset_size
#         self.batch_size = batch_size
#         self.device = device

#     def query(self, model, dataset, unlabeled_indices, acq_size, **kwargs):
#         if not hasattr(model, 'get_grad_embedding'):
#             raise ValueError('The method `get_grad_embedding` is mandatory to use badge sampling.')

#         if self.subset_size:
#             unlabeled_indices = self.rng.sample(unlabeled_indices, k=self.subset_size)
        
#         if "collator" in list(kwargs.keys()):
#             unlabeled_dataloader = DataLoader(
#                 dataset, 
#                 batch_size=self.batch_size*2, 
#                 collate_fn=kwargs['collator'],
#                 sampler=unlabeled_indices)
#         else:
#             unlabeled_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=unlabeled_indices)
#         del kwargs

#         embeddings = model.get_representation(unlabeled_dataloader, device=self.device)
#         chosen = kmeans_plusplus(grad_embedding.numpy(), acq_size, np_rng=self.np_rng)
#         return [unlabeled_indices[idx] for idx in chosen]
    

