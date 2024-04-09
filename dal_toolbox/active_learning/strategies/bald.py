# Ref: https://github.com/scikit-activeml/scikit-activeml/blob/master/skactiveml/pool/_batch_bald.py
import torch
import numpy as np
from .uncertainty import UncertaintySampling
from ..data import ActiveLearningDataModule
from ...models.utils.base import BaseModule
from ...metrics import entropy_from_logits, entropy_from_probas, ensemble_log_softmax, ensemble_entropy_from_logits


class BALDSampling(UncertaintySampling):
    def get_utilities(self, logits):
        if logits.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {logits.shape}")
        scores = self.bald_score(logits)
        return scores

    def bald_score(self, logits):
        ensemble_entropy = ensemble_entropy_from_logits(logits)
        mean_entropy = entropy_from_probas(logits.softmax(-1)).mean(dim=1)
        score = ensemble_entropy - mean_entropy
        return score


class BatchBALDSampling(UncertaintySampling):
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
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader)  # , **forward_kwargs)
        scores = self.get_utilities(logits, acq_size)
        indices = scores.argmax(dim=-1)

        actual_indices = [unlabeled_indices[i] for i in indices]
        if return_utilities:
            return actual_indices, scores
        return actual_indices

    def get_utilities(self, logits, acq_size):
        if logits.ndim != 3:
            raise ValueError(f"Input probas tensor must be 3-dimensional, got shape {logits.shape}")
        scores = self.batch_bald_score(logits, acq_size)
        return scores

    def batch_bald_score(self, logits, acq_size):
        logits = logits.cpu()
        N, K, C = logits.shape

        log_probas = torch.log_softmax(logits, dim=-1)
        conditional_entropies = _compute_conditional_entropy(log_probas)

        batch_joint_entropy = _DynamicJointEntropy(K, acq_size - 1, K, C, self.rng)
        scores = torch.zeros((acq_size, N))
        query_indices = []

        for i in range(acq_size):
            if i > 0:
                latest_index = query_indices[-1]
                batch_joint_entropy.add_variables(log_probas[latest_index: latest_index + 1])

            shared_conditinal_entropies = conditional_entropies[query_indices].sum()

            scores[i] = batch_joint_entropy.compute_batch(log_probas, output_entropies_B=scores[i])
            scores[i] -= conditional_entropies + shared_conditinal_entropies
            scores[i, query_indices] = -torch.inf

            query_idx = torch.argmax(scores[i])
            query_indices.append(query_idx.item())
        return scores


def _compute_conditional_entropy(log_probs_N_K_C):
    num_mc_samples = log_probs_N_K_C.size(1)
    nats_N_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)
    entropies_N = - torch.sum(nats_N_K_C, axis=(1, 2)) / num_mc_samples
    return entropies_N


def _gather_expand(data, axis, index):
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[axis] = data.shape[axis]

    new_index_shape = list(max_shape)
    new_index_shape[axis] = index.shape[axis]

    data = data.expand(new_data_shape)  # np.broadcast_to(data, new_data_shape)
    index = index.expand(new_index_shape)  # np.broadcast_to(index, new_index_shape)

    return torch.take_along_dim(data, index, dim=axis)


class _DynamicJointEntropy:
    def __init__(self, M, max_N, K, C, random_state):
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = _ExactJointEntropy.empty(K)
        self.log_probs_max_N_K_C = torch.empty((max_N, K, C))

        self.random_state = random_state

    def add_variables(self, log_probs_N_K_C):
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]

        self.log_probs_max_N_K_C[self.N: self.N + add_N] = log_probs_N_K_C
        self.N += add_N

        num_exact_samples = C**self.N
        if num_exact_samples > self.M:
            self.inner = _SampledJointEntropy.sample(
                torch.exp(self.log_probs_max_N_K_C[:self.N]),
                self.M,
                self.random_state,
            )
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute_batch(self, log_probs_B_K_C, output_entropies_B=None):
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        return self.inner.compute_batch(log_probs_B_K_C, output_entropies_B)


def _batch_multi_choices(probs_b_C, M, random_state):
    """
    probs_b_C: Ni... x C

    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
    B = probs_B_C.shape[0]
    C = probs_B_C.shape[1]

    # samples: Ni... x draw_per_xx
    choices = [
        random_state.choice(C, size=M, p=probs_B_C[b].numpy(), replace=True)
        for b in range(B)
    ]
    choices = np.array(choices, dtype=int)
    choices = torch.from_numpy(choices)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M


class _SampledJointEntropy:
    """Random variables (all with the same # of categories $C$) can be added via `_SampledJointEntropy.add_variables`.

    `_SampledJointEntropy.compute` computes the joint entropy.

    `_SampledJointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn.
    """

    def __init__(self, sampled_joint_probs_M_K, random_state):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def sample(probs_N_K_C, M, random_state):
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = _batch_multi_choices(probs_N_K_C, S, random_state)

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = _gather_expand(
            expanded_probs_N_K_1_C, axis=-1, index=expanded_choices_N_1_K_S
        )
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S+1e-9), axis=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.T
        return _SampledJointEntropy(samples_M_K, random_state)

    def compute_batch(self, log_probs_B_K_C, output_entropies_B=None):
        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        b = log_probs_B_K_C.shape[0]

        probs_b_M_C = torch.empty((b, M, C))
        for i in range(b):
            probs_b_M_C[i] = torch.matmul(self.sampled_joint_probs_M_K, torch.exp(log_probs_B_K_C[i]))
        probs_b_M_C /= K

        q_1_M_1 = self.sampled_joint_probs_M_K.mean(axis=1, keepdim=True)[None]

        output_entropies_B = (torch.sum(-torch.log(probs_b_M_C+1e-9) * probs_b_M_C / q_1_M_1, axis=(1, 2)) / M)

        return output_entropies_B


class _ExactJointEntropy:
    def __init__(self, joint_probs_M_K):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K):
        return _ExactJointEntropy(torch.ones((1, K)))

    def add_variables(self, log_probs_N_K_C):
        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.T[:, :, None]

        probs_N_K_C = torch.exp(log_probs_N_K_C)

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :]
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).T
        return self

    def compute_batch(self, log_probs_B_K_C, output_entropies_B=None):
        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        probs_b_K_C = torch.exp(log_probs_B_K_C)
        b = probs_b_K_C.shape[0]
        probs_b_M_C = torch.empty((b, M, C))
        for i in range(b):
            probs_b_M_C[i] = torch.matmul(self.joint_probs_M_K, probs_b_K_C[i])
        probs_b_M_C /= K

        output_entropies_B = torch.sum(-torch.log(probs_b_M_C+1e-9) * probs_b_M_C, axis=(1, 2))

        return output_entropies_B
