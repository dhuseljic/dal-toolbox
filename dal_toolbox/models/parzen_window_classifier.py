import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_kernels
from sklearn.utils import check_random_state, check_array


def kernels(X, Y, metric, **kwargs):
    metric = str(metric)
    if metric == 'rbf':
        gamma = kwargs.pop('gamma')
        return pairwise_kernels(X=X, Y=Y, metric=metric, gamma=gamma)
    elif metric == 'cosine':
        return pairwise_kernels(X=X, Y=Y, metric=metric)
    elif metric == 'categorical':
        gamma = kwargs.pop('gamma')
        return np.exp(-gamma * cdist(XA=X, XB=Y, metric='hamming'))


class PWC(BaseEstimator, ClassifierMixin):
    """PWC

    The Parzen window classifier (PWC) [1] is a simple and probabilistic classifier. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    n_classes: int,
        This parameter indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.

    Attributes
    ----------
    n_classes: int,
        This parameters indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.
    X: array-like, shape (n_samples, n_features)
        The sample matrix X is the feature matrix representing the samples.
    y: array-like, shape (n_samples) or (n_samples, n_outputs)
        It contains the class labels of the training samples.
        The number of class labels may be variable for the samples.
    Z: array-like, shape (n_samples, n_classes)
        The class labels are represented by counting vectors. An entry Z[i,j] indicates how many class labels of class j
        were provided for training sample i.

    References
    ----------
    [1] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, n_classes, metric='rbf', n_neighbors=None, random_state=42, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 0:
            raise ValueError("The parameter 'n_classes' must be a positive integer.")
        self.metric_ = str(metric)
        self.n_neighbors_ = int(n_neighbors) if n_neighbors is not None else n_neighbors
        if self.n_neighbors_ is not None and self.n_neighbors_ <= 0:
            raise ValueError("The parameter 'n_neighbors' must be a positive integer.")
        self.random_state_ = check_random_state(random_state)
        self.kwargs_ = kwargs
        self.X_ = None
        self.y_ = None
        self.Z_ = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where missing labels are
            represented by np.nan.

        Returns
        -------
        self: PWC,
            The PWC is fitted on the training data.
        """
        if np.size(X) > 0:
            self.X_ = check_array(X)
            self.y_ = check_array(y, ensure_2d=False, force_all_finite=False).astype(int)

            # convert labels to count vectors
            self.Z_ = np.zeros((np.size(X, 0), self.n_classes_))
            for i in range(np.size(self.Z_, 0)):
                self.Z_[i, self.y_[i]] += 1

        return self

    def predict_proba(self, X, **kwargs):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.
        C: array-like, shape (n_classes, n_classes)
            Classification cost matrix.

        Returns
        -------
        P:  array-like, shape (t_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # if normalize is false, the probabilities are frequency estimates
        normalize = kwargs.pop('normalize', True)

        # no training data -> random prediction
        if self.X_ is None or np.size(self.X_, 0) == 0:
            if normalize:
                return np.full((np.size(X, 0), self.n_classes_), 1. / self.n_classes_)
            else:
                return np.zeros((np.size(X, 0), self.n_classes_))

        # calculating metric matrix
        if self.metric_ == 'precomputed':
            K = X
        else:
            K = kernels(X, self.X_, metric=self.metric_, **self.kwargs_)

        if self.n_neighbors_ is None:
            # calculating labeling frequency estimates
            P = K @ self.Z_
        else:
            if np.size(self.X_, 0) < self.n_neighbors_:
                n_neighbors = np.size(self.X_, 0)
            else:
                n_neighbors = self.n_neighbors_
            indices = np.argpartition(K, -n_neighbors, axis=1)[:, -n_neighbors:]
            P = np.empty((np.size(X, 0), self.n_classes_))
            for i in range(np.size(X, 0)):
                P[i, :] = K[i, indices[i]] @ self.Z_[indices[i], :]

        if normalize:
            # normalizing probabilities of each sample
            normalizer = np.sum(P, axis=1)
            P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
            P[normalizer == 0, :] = [1 / self.n_classes_] * self.n_classes_

        return P

    def predict(self, X, **kwargs):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y:  array-like, shape = [n_samples]
            Predicted class labels class.
        """
        P = self.predict_proba(X, normalize=True)
        return self._rand_arg_max(P, axis=1)

    def reset(self):
        """
        Reset fitted parameters.
        """
        self.X_ = None
        self.y_ = None
        self.Z_ = None
        self.random_state_ = self.random_state_

    def _rand_arg_max(self, arr, axis=1):
        """
        Returns index of maximal element per given axis. In case of a tie, the index is chosen randomly.
        
        Parameters
        ----------
        arr: array-like
            Array whose minimal elements' indices are determined.
        axis: int
            Indices of minimal elements are determined along this axis.

        Returns
        -------
        min_indices: array-like
            Indices of maximal elements.
        """
        arr_max = arr.max(axis, keepdims=True)
        tmp = self.random_state_.uniform(low=1, high=2, size=arr.shape) * (arr == arr_max)
        return tmp.argmax(axis)

    def get_params(self, deep=True):
        return {'n_classes': self.n_classes_, 'metric': self.metric_, 'n_neighbors': self.n_neighbors_,
                'random_state': self.random_state_, **self.kwargs_}
