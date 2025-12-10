import torch
import torch.nn as nn
import numpy as np
import lightning as L

from torch.utils.data import Subset, RandomSampler, DataLoader, Dataset

from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..utils import setup_rng


class ActiveLearningDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for active learning workflows.

    This module manages datasets for training, validation, testing, and querying in an active learning context. 
    It also handles labeled and unlabeled data pools and provides dataloaders for labeled and unlabeled data.

    Args:
        train_dataset (Dataset): The dataset used for training.
        query_dataset (Dataset, optional): The dataset used for querying unlabeled data. If None, the train_dataset will be used. Defaults to None.
        val_dataset (Dataset, optional): The dataset used for validation. Defaults to None.
        test_dataset (Dataset, optional): The dataset used for testing. Defaults to None.
        train_batch_size (int, optional): The batch size for the training DataLoader. Defaults to 64.
        predict_batch_size (int, optional): The batch size for the unlabeled/query and prediction DataLoader. Defaults to 256.
        seed (int, optional): The seed for random number generation. Defaults to None.
        collator (callable, optional): A custom collator function for the DataLoader. Defaults to None.

    Attributes:
        train_dataset (Dataset): The dataset for training.
        query_dataset (QueryDataset): The dataset for querying in active learning.
        val_dataset (Dataset, optional): The dataset for validation.
        test_dataset (Dataset, optional): The dataset for testing.
        train_batch_size (int): Batch size for the training DataLoader.
        predict_batch_size (int): Batch size for prediction and unlabeled DataLoader.
        collator (callable, optional): Collator function for the DataLoader.
        rng (np.random.RandomState): Random number generator for sampling.
        unlabeled_indices (list): List of indices of unlabeled data.
        labeled_indices (list): List of indices of labeled data.
    """

    def __init__(
            self,
            train_dataset: Dataset,
            query_dataset: Dataset = None,
            val_dataset: Dataset = None,
            test_dataset: Dataset = None,
            train_batch_size: int = 64,
            predict_batch_size: int = 256,
            seed: int = None,
            collator=None
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.query_dataset = QueryDataset(
            query_dataset) if query_dataset else QueryDataset(dataset=train_dataset)
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.collator = collator

        if query_dataset is None:
            rank_zero_warn('Using train_dataset for queries. Ensure that there are no augmentations used.')

        if self.val_dataset is not None:
            self.val_dataloader = lambda: DataLoader(
                self.val_dataset, batch_size=predict_batch_size, shuffle=False)

        if self.test_dataset is not None:
            self.test_dataloader = lambda: DataLoader(
                self.test_dataset, batch_size=predict_batch_size, shuffle=False)

        self.rng = setup_rng(seed)
        self.unlabeled_indices = list(range(len(self.train_dataset)))
        self.labeled_indices = []

    def train_dataloader(self):
        if len(self.labeled_indices) == 0:
            raise ValueError('No instances labeled yet. Initialize the labeled pool first.')

        labeled_dataset = Subset(self.train_dataset, indices=self.labeled_indices)
        sampler = RandomSampler(labeled_dataset)
        drop_last = (len(sampler) > self.train_batch_size)
        train_loader = DataLoader(
            labeled_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            drop_last=drop_last
        )
        return train_loader

    def unlabeled_dataloader(self, subset_size=None):
        """Returns a dataloader for the unlabeled pool where instances are not augmentated."""
        unlabeled_indices = self._subsample_indices(self.unlabeled_indices, subset_size)
        loader = DataLoader(self.query_dataset, batch_size=self.predict_batch_size,
                            sampler=unlabeled_indices, collate_fn=self.collator)
        return loader, unlabeled_indices

    def labeled_dataloader(self, subset_size=None):
        """Returns a dataloader for the labeled pool where instances are not augmentated."""
        labeled_indices = self._subsample_indices(self.labeled_indices, subset_size)
        loader = DataLoader(self.query_dataset, batch_size=self.predict_batch_size,
                            sampler=labeled_indices, collate_fn=self.collator)
        return loader, labeled_indices

    def custom_dataloader(self, indices: list, train: bool = False, custom_labels=None, custom_batch_size=None):
        if train:
            custom_dataset = Subset(self.train_dataset, indices=indices)
            sampler = RandomSampler(custom_dataset)
            batch_size = self.train_batch_size
            drop_last = (len(indices) > self.train_batch_size)
        else:
            custom_dataset = Subset(self.query_dataset, indices=indices)
            sampler = None
            batch_size = self.predict_batch_size
            drop_last = False

        if custom_labels is not None:
            custom_dataset = RelabeledDataset(custom_dataset, custom_labels)
        if custom_batch_size is not None:
            batch_size = custom_batch_size

        loader = DataLoader(
            custom_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            drop_last=drop_last
        )
        return loader

    def state_dict(self):
        state_dict = {
            'labeled_indices': self.labeled_indices,
            'unlabeled_indices': self.unlabeled_indices,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.labeled_indices = state_dict['labeled_indices']
        self.unlabeled_indices = state_dict['unlabeled_indices']

    def update_annotations(self, buy_idx: list):
        """
            Updates the labeled pool with newly annotated instances.

            Args:
                buy_idx (list): List of indices which identify samples of the unlabeled pool that should be
                                transfered to the labeld pool.
        """
        # Check if buy_idx has duplicate indices
        if len(buy_idx) != len(set(buy_idx)):
            raise ValueError('The `buy_idx` has duplicate annotations.')
        # Check if annotation already in labeled indices
        if np.any(np.isin(buy_idx, self.labeled_indices)):
            raise ValueError('Some `buy_idx` annotations are already in labeled indices.')
        self.labeled_indices = list_union(self.labeled_indices, buy_idx)
        self.unlabeled_indices = list_diff(self.unlabeled_indices, buy_idx)

    def random_init(self, n_samples: int, class_balanced: bool = False):
        """Randomly annotates instances from the unlabeled pool and adds them to the labeled one.

            Args:
                n_samples (int): Size of the initial labeld pool.    
                class_balanced (bool): Whether to use an class balanced initialization.
        """
        if len(self.labeled_indices) != 0:
            raise ValueError('Pools already initialized.')

        if class_balanced:
            # TODO(dhuseljic): problematic when data gets loaded e.g., imagenet
            classes = torch.Tensor([self.query_dataset[idx][1] for idx in self.unlabeled_indices]).long()
            classes_unique = classes.unique()
            n_classes = len(classes_unique)
            n_samples_per_class = n_samples // n_classes

            indices = []
            for label in classes_unique:
                unlabeled_indices_lbl = (classes == label).nonzero().squeeze()
                indices_lbl = self.rng.choice(unlabeled_indices_lbl.tolist(),
                                              size=n_samples_per_class, replace=False)
                indices_lbl = indices_lbl.tolist()
                indices.extend(indices_lbl)
        else:
            indices = self.rng.choice(self.unlabeled_indices, size=n_samples, replace=False)
            indices = indices.tolist()
        self.update_annotations(indices)

    def _subsample_indices(self, indices, subset_size):
        if subset_size is None:
            return indices
        
        if isinstance(subset_size, float):
            if not 0 < subset_size <= 1:
                raise ValueError(f"subset_size as float must be in (0, 1], got {subset_size}")
            actual_size = int(len(indices) * subset_size)
        else:
            actual_size = subset_size
        
        actual_size = min(len(indices), actual_size)
        subsampled_indices = self.rng.choice(indices, size=actual_size, replace=False)
        return subsampled_indices.tolist()

    def diverse_dense_init(self, n_samples: int, model: nn.Module = None, num_neighbors=20):
        if len(self.labeled_indices) != 0:
            raise ValueError('Pools already initialized.')

        # Get features
        dataloader, unlabeled_indices = self.unlabeled_dataloader()
        features = []
        for batch in dataloader:
            if model is None:
                feature = batch[0]
            else:
                feature = model(batch[0])
            features.append(feature)
        features = torch.cat(features)
        features = features.numpy()

        kmeans = KMeans(n_clusters=n_samples, n_init='auto')
        clusters = kmeans.fit_predict(features)

        indices = []
        for cluster in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            features_cluster = features[cluster_indices]

            neighbors = NearestNeighbors(n_neighbors=num_neighbors+1, metric='sqeuclidean', n_jobs=-1)
            neighbors.fit(features_cluster)
            distances, _ = neighbors.kneighbors(features_cluster)
            idx = distances.sum(-1).argmin()
            indices.append(cluster_indices[idx])
        self.update_annotations(indices)

    def update_datasets(self, train_dataset, query_dataset=None, val_dataset=None, test_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.query_dataset = QueryDataset(
            query_dataset) if query_dataset else QueryDataset(dataset=train_dataset)


class QueryDataset(Dataset):
    """A helper class which returns also the index along with the instances and targets."""
    # problem with dictionary output of dataset

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        if isinstance(data, dict):
            instance = {
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask'],
                'label': data['label'],
                'index': index
            }
            return instance
        else:
            instance, target = self.dataset.__getitem__(index)
            return instance, target, index

    def __len__(self):
        return len(self.dataset)


class RelabeledDataset(Dataset):
    """
    A custom dataset wrapper that replaces the labels of the original dataset with new labels.

    Args:
        dataset (Dataset): The original dataset containing the data and old labels.
        labels (Tensor or list): A tensor or list of new labels that will replace the original labels.

    Example:
        >>> original_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        >>> new_labels = torch.randint(0, 10, (len(original_dataset),))  # Example random labels
        >>> dataset = LabelReplacedDataset(original_dataset, new_labels)
    """

    def __init__(self, dataset, labels):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        if len(dataset) != len(labels):
            raise ValueError('The labels should have the same length as the dataset.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        batch = list(batch)
        batch[1] = self.labels[idx]
        return batch


def list_union(a: list, b: list):
    return list(set(a).union(set(b)))


def list_diff(a: list, b: list):
    return list(set(a).difference(set(b)))
