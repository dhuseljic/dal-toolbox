import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, check_array
from torch import nn

from dal_toolbox.models.utils.base import BaseModule
from dal_toolbox.utils import kernels


class PWCLightning(BaseModule):
    """
    Wrapper for PWC to work with lightning.
    """

    def __init__(self, n_classes, random_state, kernel_params, **kwargs):
        # TODO (ynagel) Maybe find a better solution for this
        metric = kernel_params["kernel"]["name"]
        gamma = kernel_params["kernel"]["gamma"]
        n_neighbors = kernel_params["n_neighbors"]
        model = PWC(n_classes, metric, n_neighbors, random_state, **{"gamma": gamma})
        super().__init__(model, **kwargs)
        self.fake_layer = nn.Parameter(torch.Tensor([1.0]))  # This is a fake parameter to satisfy any optimizers

        self.training_inputs = []
        self.training_targets = []

    def training_step(self, batch):
        inputs, targets = batch[0], batch[1]

        self.training_inputs.append(inputs)
        self.training_targets.append(targets)

    def validation_step(self, batch, batch_idx):
        X, y = batch[0].numpy(force=True), batch[1]

        y_proba = self.model.predict_proba(X)
        self.log_val_metrics(torch.from_numpy(y_proba), y)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch > 0:
            pass

        X = torch.cat(self.training_inputs).numpy(force=True)
        y = torch.cat(self.training_targets)

        self.model.fit(X, y.numpy(force=True))

        y_proba = self.model.predict_proba(X)
        self.log_train_metrics(torch.from_numpy(y_proba), y)

        self.training_inputs = []
        self.training_targets = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch

        inputs = inputs.numpy(force=True)
        probas = self.model.predict_proba(inputs)

        return torch.Tensor(probas), targets

    def reset_states(self, reset_model_parameters=True):
        self.model.reset()


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

        self.state_dict = lambda: self.get_params(deep=True)

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

    def get_logits(self, dataloader, device):
        inputs = torch.cat([b[0] for b in dataloader]).numpy(force=True)
        logits = self.predict_proba(inputs)
        return torch.from_numpy(logits)

    def get_probas(self, dataloader, device):
        inputs = torch.cat([b[0] for b in dataloader]).numpy(force=True)
        probas = self.predict_proba(inputs)
        return torch.from_numpy(probas)

    def get_representations(self, dataloader, device, return_labels=False):
        all_features = []
        all_labels = []
        for batch in dataloader:
            features = batch[0]
            labels = batch[1]
            all_features.append(features.cpu())
            all_labels.append(labels)
        features = torch.cat(all_features)

        if return_labels:
            labels = torch.cat(all_labels)
            return features, labels
        return features

    def get_grad_representations(self, dataloader, device):
        embedding = []
        for batch in dataloader:
            inputs = batch[0]
            in_dimension = inputs.shape[-1]
            embedding_batch = torch.empty([len(inputs), in_dimension * self.n_classes_])
            features = inputs.cpu()

            probas = self.predict_proba(inputs, normalize=True)
            max_indices = probas.argmax(-1)

            for n in range(len(inputs)):
                for c in range(self.n_classes_):
                    if c == max_indices[n]:
                        embedding_batch[n, in_dimension * c: in_dimension * (c + 1)] = \
                            features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, in_dimension * c: in_dimension * (c + 1)] = \
                            features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding
