import torch
import numpy as np
from .query import Query

from copy import copy as copy
from copy import deepcopy as deepcopy
from torch.utils.data import TensorDataset, DataLoader
from rich.progress import track


class BaitSampling(Query):
    def __init__(self,
                 lmb=1,
                 grad_approx=True,
                 grad_k=1,
                 fisher_approx='full',
                 fisher_k=10,
                 fisher_batch_size=32,
                 select='forward_backward',
                 device='cpu',
                 subset_size=None):
        super().__init__()
        self.subset_size = subset_size
        self.lmb = lmb

        self.grad_approx = grad_approx
        self.grad_k = grad_k
        self.fisher_approx = fisher_approx
        self.fisher_k = fisher_k
        self.fisher_batch_size = fisher_batch_size
        self.select = select
        self.device = device

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        if self.fisher_approx == 'full':
            repr_unlabeled = model.get_exp_grad_representations(
                unlabeled_dataloader, grad_approx=self.grad_approx, device=self.device)
            repr_labeled = model.get_exp_grad_representations(
                labeled_dataloader, grad_approx=self.grad_approx, device=self.device)
            repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)
        elif self.fisher_approx == 'topk':
            repr_unlabeled = model.get_topk_grad_representations(
                unlabeled_dataloader, grad_approx=self.grad_approx, k=self.fisher_k, device=self.device)
            repr_labeled = model.get_topk_grad_representations(
                labeled_dataloader, grad_approx=self.grad_approx, k=self.fisher_k, device=self.device)
            repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)
        elif self.fisher_approx == 'max_pred':
            repr_unlabeled = model.get_grad_representations(
                unlabeled_dataloader, grad_approx=self.grad_approx, device=self.device)
            repr_labeled = model.get_grad_representations(
                labeled_dataloader, grad_approx=self.grad_approx, device=self.device)
            repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)

            repr_unlabeled = repr_unlabeled[:, None]
            repr_labeled = repr_labeled[:, None]
            repr_all = repr_all[:, None]
        else:
            raise NotImplementedError()

        fisher_all = torch.zeros(repr_all.size(-1), repr_all.size(-1)).to(self.device)
        dl = DataLoader(TensorDataset(repr_all), batch_size=self.fisher_batch_size, shuffle=False)
        for batch in dl:
            repr_batch = batch[0].to(self.device)
            term = torch.matmul(repr_batch.transpose(1, 2), repr_batch) / len(repr_batch)
            fisher_all += torch.sum(term, dim=0)

        fisher_labeled = torch.zeros(repr_all.size(-1), repr_all.size(-1)).to(self.device)
        dl = DataLoader(TensorDataset(repr_labeled), batch_size=self.fisher_batch_size, shuffle=False)
        for batch in dl:
            repr_batch = batch[0].to(self.device)
            term = torch.matmul(repr_batch.transpose(1, 2), repr_batch) / len(repr_batch)
            fisher_labeled += torch.sum(term, dim=0)

        if self.select == 'forward_backward':
            chosen = select_forward_backward(repr_unlabeled, acq_size, fisher_all, fisher_labeled,
                                             lmb=self.lmb, num_labeled=len(labeled_indices), device=self.device)
        elif self.select == 'foward':
            raise NotImplementedError()
        elif self.select == 'topk':
            chosen = select_topk(repr_unlabeled, acq_size, fisher_all, fisher_labeled,
                                 lmb=self.lmb, num_labeled=len(labeled_indices))
        else:
            raise NotImplementedError()

        return [unlabeled_indices[idx] for idx in chosen]


@torch.no_grad()
def select_forward_backward(repr_unlabeled, acq_size, fisher_all, fisher_labeled, lmb=1, num_labeled=0, device='cpu'):
    indsAll = []
    dim = repr_unlabeled.shape[-1]
    rank = repr_unlabeled.shape[-2]

    currentInv = torch.inverse(lmb * torch.eye(dim).to(device) +
                               fisher_labeled.to(device) * num_labeled / (num_labeled + acq_size))
    repr_unlabeled = repr_unlabeled * np.sqrt(acq_size / (num_labeled + acq_size))
    fisher_all = fisher_all.to(device)

    # forward selection, over-sample by 2x
    # print('forward selection...', flush=True)
    over_sample = 2
    # for i in range(int(over_sample * acq_size)):
    for i in track(range(int(over_sample * acq_size)), 'Bait: Oversampling'):
        # check trace with low-rank updates (woodbury identity)
        xt_ = repr_unlabeled.to(device)

        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
            innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher_all @ currentInv @
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
        xt_ = repr_unlabeled[ind].unsqueeze(0).to(device)
        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]
    # print(indsAll)

    # backward pruning
    # print('backward pruning...', flush=True)
    for i in track(range(len(indsAll) - acq_size), 'Bait: Backward pruning'):
        # for i in range(len(indsAll) - acq_size):

        # select index for removal
        xt_ = repr_unlabeled[indsAll].to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @
                                 currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher_all @ currentInv @
                                  xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

        # low-rank update (woodbury identity)
        xt_ = repr_unlabeled[indsAll[delInd]].unsqueeze(0).to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @
                                 currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    # print(indsAll)
    return indsAll


def select_topk(repr_unlabeled, acq_size, fisher_all, fisher_labeled, lmb, num_labeled):
    device = fisher_all.device

    # Efficient computation of the objective (trace rotation & Woodbury identity)
    repr_unlabeled = repr_unlabeled.to(device)
    dim = repr_unlabeled.size(-1)
    rank = repr_unlabeled.size(-2)

    fisher_labeled = fisher_labeled * num_labeled / (num_labeled + acq_size)
    M_0 = lmb * torch.eye(dim, device=device) + fisher_labeled
    M_0_inv = torch.inverse(M_0)

    # repr_unlabeled = repr_unlabeled * np.sqrt(acq_size / (num_labeled + acq_size))
    A = torch.inverse(torch.eye(rank, device=device) + repr_unlabeled @
                      M_0_inv @ repr_unlabeled.transpose(1, 2))
    tmp = repr_unlabeled @ M_0_inv @ fisher_all @ M_0_inv @ repr_unlabeled.transpose(1, 2) @ A
    scores = torch.diagonal(tmp, dim1=-2, dim2=-1).sum(-1)
    chosen = (scores.topk(acq_size).indices)
    return chosen
