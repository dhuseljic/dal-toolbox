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
                 expectation_topk=None,
                 normalize_top_probas=True,
                 fisher_approximation='full',
                 num_grad_samples=None,
                 grad_likelihood='cross_entropy',
                 grad_selection='magnitude',
                 select='forward_backward',
                 fisher_batch_size=32,
                 device='cpu',
                 subset_size=None):
        super().__init__()
        self.subset_size = subset_size
        self.lmb = lmb

        self.expectation_topk = expectation_topk
        self.normalize_top_probas = normalize_top_probas
        self.fisher_approximation = fisher_approximation
        self.num_grad_samples = num_grad_samples
        self.grad_likelihood = grad_likelihood
        self.grad_selection = grad_selection
        self.fisher_batch_size = fisher_batch_size
        self.select = select
        self.device = device

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        if self.expectation_topk is None:
            repr_unlabeled = model.get_exp_grad_representations(
                unlabeled_dataloader,
                grad_likelihood=self.grad_likelihood,
                device=self.device
            )
            repr_labeled = model.get_exp_grad_representations(
                labeled_dataloader,
                grad_likelihood=self.grad_likelihood,
                device=self.device
            )
            repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)
        else:
            repr_unlabeled = model.get_topk_grad_representations(
                unlabeled_dataloader,
                topk=self.expectation_topk,
                grad_likelihood=self.grad_likelihood,
                normalize_top_probas=self.normalize_top_probas,
                device=self.device
            )
            repr_labeled = model.get_topk_grad_representations(
                labeled_dataloader,
                topk=self.expectation_topk,
                grad_likelihood=self.grad_likelihood,
                normalize_top_probas=self.normalize_top_probas,
                device=self.device
            )
            repr_all = torch.cat((repr_unlabeled, repr_labeled), dim=0)

        if self.num_grad_samples is not None:
            if self.grad_selection == 'random':
                indices = torch.randperm(repr_all.size(-1))[:self.num_grad_samples]
                grad_indices = indices[-self.num_grad_samples:]
            elif self.grad_selection == 'std':
                indices = repr_all.mean(0).std(0).argsort()
                grad_indices = indices[-self.num_grad_samples:]
            elif self.grad_selection == 'magnitude':
                indices = torch.abs(repr_all).sum(dim=(0, 1)).argsort()
                grad_indices = indices[-self.num_grad_samples:]
            else:
                raise NotImplementedError()
            repr_unlabeled = repr_unlabeled[:, :, grad_indices]
            repr_labeled = repr_labeled[:, :, grad_indices]
            repr_all = repr_all[:, :, grad_indices]

        if self.fisher_approximation == 'full':
            fisher_dim = (repr_all.size(-1), repr_all.size(-1))
        elif self.fisher_approximation == 'block_diag':
            fisher_dim = (10, 384, 384)
        elif self.fisher_approximation == 'diag':
            fisher_dim = (repr_all.size(-1),)
        else:
            raise NotImplementedError()

        fisher_all = torch.zeros(fisher_dim).to(self.device)
        dl = DataLoader(TensorDataset(repr_all), batch_size=self.fisher_batch_size, shuffle=False)
        for batch in track(dl, "Bait: Computing fisher..", disable=True):
            repr_batch = batch[0].to(self.device)
            if self.fisher_approximation == 'full':
                term = torch.matmul(repr_batch.transpose(1, 2), repr_batch)
                fisher_all += torch.mean(term, dim=0)
            elif self.fisher_approximation == 'block_diag':
                repr_batch = repr_batch.view(-1, 10, 10, 384)
                term = torch.einsum('nkhd,mkhe->hde', repr_batch, repr_batch) 
                fisher_all += term / len(repr_batch)
            elif self.fisher_approximation == 'diag':
                term = torch.mean(torch.sum(repr_batch**2, dim=1), dim=0)
                fisher_all += term
            else:
                raise NotImplementedError()

        fisher_labeled = torch.zeros(fisher_dim).to(self.device)
        dl = DataLoader(TensorDataset(repr_labeled), batch_size=self.fisher_batch_size, shuffle=False)
        for batch in track(dl, "Bait: Computing fisher..", disable=True):
            repr_batch = batch[0].to(self.device)
            if self.fisher_approximation == 'full':
                term = torch.matmul(repr_batch.transpose(1, 2), repr_batch)
                fisher_labeled += torch.mean(term, dim=0)
            elif self.fisher_approximation == 'block_diag':
                repr_batch = repr_batch.view(-1, 10, 10, 384)
                term = torch.einsum('nkhd,mkhe->hde', repr_batch, repr_batch) 
                fisher_labeled += term / len(repr_batch)
            elif self.fisher_approximation == 'diag':
                term = torch.mean(torch.sum(repr_batch**2, dim=1), dim=0)
                fisher_labeled += term
            else:
                raise NotImplementedError()

        # Clear memory for selection
        del repr_batch
        del term
        fisher_all = fisher_all.cpu()
        fisher_labeled = fisher_labeled.cpu()
        torch.cuda.empty_cache()

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
    is_diag = (fisher_all.ndim == 1)
    is_block_diag = (fisher_all.ndim == 3)

    indsAll = []
    dim = repr_unlabeled.shape[-1]
    rank = repr_unlabeled.shape[-2]

    if is_diag:
        currentInv = 1 / (lmb + fisher_labeled * num_labeled / (num_labeled + acq_size))
    elif is_block_diag:
        currentInv = torch.inverse(lmb + fisher_labeled * num_labeled / (num_labeled + acq_size))
    else:
        inv_device = 'cpu' if fisher_labeled.size(0) > 10_000 else device
        currentInv = torch.inverse(lmb * torch.eye(dim, device=inv_device) + fisher_labeled.to(inv_device) *
                                   num_labeled / (num_labeled + acq_size))
    repr_unlabeled = repr_unlabeled * np.sqrt(acq_size / (num_labeled + acq_size))

    fisher_all = fisher_all.to(device)
    currentInv = currentInv.to(device)
    # forward selection, over-sample by 2x
    # print('forward selection...', flush=True)
    over_sample = 2
    # for i in range(int(over_sample * acq_size)):
    for i in track(range(int(over_sample * acq_size)), 'Bait: Oversampling', disable=True):
        # check trace with low-rank updates (woodbury identity)
        xt_ = repr_unlabeled.to(device)

        if is_diag:
            innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ *
                                     currentInv @ xt_.transpose(1, 2)).detach()
        elif is_block_diag:
            raise NotImplementedError()
        else:
            innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2))
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
            innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        if is_diag:
            traceEst = torch.diagonal(xt_ * currentInv * fisher_all * currentInv @
                                      xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        elif is_block_diag:
            raise NotImplementedError()
        else:
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher_all @ currentInv @
                                      xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break
        indsAll.append(ind)
        # print(i, ind, traceEst[ind], flush=True)

        # commit to a low-rank update
        del xt_
        torch.cuda.empty_cache()
        xt_ = repr_unlabeled[ind].unsqueeze(0).to(device)
        if is_diag:
            innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ * currentInv @ xt_.transpose(1, 2))
            currentInv = (currentInv - torch.diag(((xt_*currentInv).transpose(1, 2)
                          @ innerInv @ (xt_*currentInv))[0]))
        elif is_block_diag:
            raise NotImplementedError()
        else:
            # xt_.cpu() @ currentInv.cpu() @ xt_.transpose(1, 2).cpu()
            innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2))
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv)[0]
        del xt_
        del innerInv
        torch.cuda.empty_cache()

    # print(indsAll)

    # backward pruning
    # print('backward pruning...', flush=True)
    for i in track(range(len(indsAll) - acq_size), 'Bait: Backward pruning', disable=True):
        # for i in range(len(indsAll) - acq_size):

        # select index for removal
        xt_ = repr_unlabeled[indsAll].to(device)
        if is_diag:
            innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ * currentInv @ xt_.transpose(1, 2))
            traceEst = torch.diagonal(xt_ * currentInv * fisher_all * currentInv @
                                      xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        elif is_block_diag:
            raise NotImplementedError()
        else:
            innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2))
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher_all @ currentInv @
                                      xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

        # low-rank update (woodbury identity)
        xt_ = repr_unlabeled[indsAll[delInd]].unsqueeze(0).to(device)

        if is_diag:
            innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ * currentInv @ xt_.transpose(1, 2))
            currentInv = (currentInv - torch.diag(((xt_*currentInv).transpose(1, 2)
                          @ innerInv @ (xt_*currentInv))[0]))
        elif is_block_diag:
            raise NotImplementedError()
        else:
            innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2))
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv)[0]

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
