import torch
import numpy as np
from .query import Query

import gc
from copy import copy as copy
from copy import deepcopy as deepcopy
from torch.utils.data import TensorDataset, DataLoader


class BaitSampling(Query):
    def __init__(self, lmb, fisher_batch_size=32, device='cpu', subset_size=None):
        super().__init__()
        self.subset_size = subset_size
        self.lmb = lmb
        self.fisher_batch_size = fisher_batch_size
        self.device = device

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        # Use all data
        repr_unlabeled = model.get_exp_grad_representations(unlabeled_dataloader)
        repr_labeled = model.get_exp_grad_representations(labeled_dataloader)
        repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)

        fisher = torch.zeros(repr_all.size(-1), repr_all.size(-1)).to(self.device)
        dl = DataLoader(TensorDataset(repr_all), batch_size=self.fisher_batch_size, shuffle=False)
        for batch in dl:
            repr_batch = batch[0].to(self.device)
            term = torch.matmul(repr_batch.transpose(1, 2), repr_batch) / len(repr_batch)
            fisher += torch.sum(term, dim=0)

        init = torch.zeros(repr_all.size(-1), repr_all.size(-1)).to(self.device)

        for batch in dl:
            repr_batch = batch[0].to(self.device)
            term = torch.matmul(repr_batch.transpose(1, 2), repr_batch) / len(repr_batch)
            init += torch.sum(term, dim=0)

        chosen = select(repr_unlabeled, acq_size, fisher, init, lamb=self.lmb, nLabeled=len(labeled_indices))
        return [unlabeled_indices[idx] for idx in chosen]


@torch.no_grad()
def select(X, K, fisher, iterates, lamb=1, nLabeled=0, device='cpu'):
    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).to(device) +
                               iterates.to(device) * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.to(device)

    # forward selection, over-sample by 2x
    # print('forward selection...', flush=True)
    over_sample = 2
    for i in range(int(over_sample * K)):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.to(device)
        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
            innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @
                                  xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        # clear out gpu memory

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        # print(i, ind, traceEst[ind], flush=True)

        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).to(device)
        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    # print('backward pruning...', flush=True)
    for i in range(len(indsAll) - K):

        # select index for removal
        xt_ = X[indsAll].to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @
                                 currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @
                                  xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @
                                 currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    return indsAll
