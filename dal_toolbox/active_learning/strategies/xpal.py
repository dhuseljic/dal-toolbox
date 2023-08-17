# https://github.com/dakot/probal/blob/master/src/query_strategies/expected_probabilistic_active_learning.py

import numpy as np
import torch
from sklearn.utils import check_array

from .query import Query


class XPAL(Query):
    """XPAL
    The expected probabilistic active learning (xPAL) strategy.
    Parameters
    ----------
    n_classes: int
        Number of classes.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    data_set: base.DataSet
        Data set containing samples and class labels.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.
    Attributes
    ----------
    n_classes_: int
        Number of classes.
    S_: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all pairs of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    alpha_c_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    alpha_x_: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    """

    def __init__(self, num_classes, S, alpha_c, alpha_x, subset_size=None, random_seed=None):
        super().__init__(random_seed)

        self.num_classes = num_classes
        self.S = S
        self.alpha_c_ = alpha_c
        self.alpha_x_ = alpha_x
        self.subset_size = subset_size

    def query(self, *, model, al_datamodule, acq_size, return_gains=False, **kwargs):
        """Compute score for each unlabeled sample. Score is to be maximized.
        Parameters
        ----------
        unlabeled_indices: array-like, shape (n_unlabeled_samples)
        Returns
        -------
        scores: array-like, shape (n_unlabeled_samples)
            Score of each unlabeled sample.
        """
        # compute frequency estimates for evaluation set (K_x) and candidate set (K_c)
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)
        labeled_loader, labeled_indices = al_datamodule.labeled_dataloader()

        if self.subset_size is None:
            S_ = self.S

            mapped_labeled_indices = labeled_indices
            mapped_unlabeled_indices = unlabeled_indices
        else:
            # Deletes entries from S, that are not part of the labeled/unlabeled subset
            existing_indices = np.concatenate([unlabeled_indices, labeled_indices])
            indices_to_remove = np.arange(self.S.shape[0])
            mask = np.equal.outer(indices_to_remove, existing_indices)
            indices_to_remove = indices_to_remove[~np.logical_or.reduce(mask, axis=1)]

            S_ = np.delete(self.S, indices_to_remove, 0)
            S_ = np.delete(S_, indices_to_remove, 1)

            # Remapping indices
            mapping = np.argsort(np.argsort(existing_indices))
            mapped_labeled_indices = mapping[len(unlabeled_indices):]
            mapped_unlabeled_indices = mapping[:len(unlabeled_indices)]

        y_labeled = []

        for batch in labeled_loader:
            y_labeled.append(batch[1])
        if len(y_labeled) > 0:
            y_labeled = torch.cat(y_labeled).tolist()

        Z = np.eye(self.num_classes)[y_labeled]
        K_x = S_[:, mapped_labeled_indices] @ Z
        K_c = K_x[mapped_unlabeled_indices]

        # calculate loss reduction for each unlabeled sample
        gains = xpal_gain(K_c=K_c, K_x=K_x, S=S_[mapped_unlabeled_indices], alpha_c=self.alpha_c_, alpha_x=self.alpha_x_)

        top_gains, indices = torch.topk(torch.Tensor(gains), acq_size)

        actual_indices = [int(unlabeled_indices[i]) for i in indices]

        if return_gains:
            return actual_indices, top_gains
        return actual_indices


def xpal_gain(K_c, K_x=None, S=None, alpha_x=1, alpha_c=1):
    """
    Computes the expected probabilistic gain.
    Parameters
    ----------
    K_c: array-like, shape (n_candidate_samples, n_classes)
        Kernel frequency estimate vectors of the candidate samples.
    K_x: array-like, shape (n_evaluation_samples, n_classes), optional (default=K_c))
        Kernel frequency estimate vectors of the evaluation samples.
    S: array-like, shape (n_candidate_samples, n_evaluation_samples), optional (default=np.eye(n_candidate_samples))
        Similarities between all pairs of candidate and evaluation samples
    alpha_x: array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the samples in the evaluation set.
        Default is 1 for all classes.
    alpha_c: float | array-like, shape (n_classes)
        Prior probabilities for the Dirichlet distribution of the candidate samples.
        Default is 1 for all classes.
    Returns
    -------
    gains: numpy.ndarray, shape (n_candidate_samples)
        Computed expected gain for each candidate sample.
    """
    # check kernel frequency estimates of candidate samples
    K_c = check_array(K_c)
    n_candidate_samples = K_c.shape[0]
    n_classes = K_c.shape[1]

    # check kernel frequency estimates of evaluation samples
    K_x = K_c if K_x is None else check_array(K_x)
    n_evaluation_samples = K_x.shape[0]
    if n_classes != K_x.shape[1]:
        raise ValueError("'K_x' and 'K_c' must have one column per class")

    # check similarity matrix
    S = np.eye(n_candidate_samples) if S is None else check_array(S)
    if S.shape[0] != n_candidate_samples or S.shape[1] != n_evaluation_samples:
        raise ValueError("'S' must have the shape (n_candidate_samples, n_evaluation_samples)")

    # check prior parameters
    if hasattr(alpha_c, "__len__") and len(alpha_c) != n_classes:
        raise ValueError("'alpha_c' must be either a float > 0 or array-like with shape (n_classes)")
    if hasattr(alpha_x, "__len__") and len(alpha_x) != n_classes:
        raise ValueError("'alpha_x' must be either a float > 0 or array-like with shape (n_classes)")

    # uniform risk matrix
    R = 1 - np.eye(n_classes)

    # model future hypothetical labels
    l_vecs = np.eye(n_classes, dtype=int)

    # compute possible risk differences
    class_vector = np.arange(n_classes, dtype=int)
    R_diff = np.array([[R[:, y_hat] - R[:, y_hat_l] for y_hat_l in class_vector] for y_hat in class_vector])

    # compute current error per evaluation sample and class
    R_x = K_x @ R

    # compute current predictions
    y_hat = np.argmin(R_x, axis=1)

    # compute required labels per class to flip decision
    with np.errstate(divide='ignore', invalid='ignore'):
        D_x = np.nanmin(np.divide(R_x - np.min(R_x, axis=1, keepdims=True), R[:, y_hat].T), axis=1)
        D_x = np.tile(D_x, (len(S), 1))  # TODO This takes up a lot of space with large datasets

    # indicates where a decision flip can be reached
    I = D_x - S < 0

    # compute normalization constants per candidate sample
    K_c_alpha_c_norm = K_c + alpha_c
    K_c_alpha_c_norm /= K_c_alpha_c_norm.sum(axis=1, keepdims=True)

    # stores gain per candidate sample
    gains = np.zeros(n_candidate_samples)

    # TODO This has to be adapted for DNNs
    # compute gain for each candidate sample
    flip_indices = np.argwhere(np.sum(I, axis=1) > 0)[:, 0]
    for ik_c in flip_indices:
        for l_idx, l_vec in enumerate(l_vecs):
            K_l = (S[ik_c, I[ik_c]] * l_vec[:, np.newaxis]).T
            K_new = K_x[I[ik_c]] + K_l
            y_hat_l = np.argmin(K_new @ R, axis=1)
            K_new += alpha_x
            K_new /= np.sum(K_new, axis=1, keepdims=True)
            gains[ik_c] += K_c_alpha_c_norm[ik_c, l_idx] * np.sum(K_new * R_diff[y_hat[I[ik_c]], y_hat_l])

    # compute average gains over evaluation samples
    gains /= n_evaluation_samples

    return gains
