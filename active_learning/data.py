import torch
import random
from torch.utils.data import Subset


class UnlabeledDataset(Subset):
    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        return x, torch.tensor(-1).long()


class FullDataset(Subset):
    def __init__(self, dataset, unlabeled_indices):
        super().__init__(dataset=dataset, indices=range(len(dataset)))
        self.dataset = dataset
        self.unlabeled_indices = unlabeled_indices

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if idx in self.unlabeled_indices:
            return x, torch.tensor(-1).long()
        return x, y


class ALDataset:
    def __init__(self, train_dataset, query_dataset):
        # Differenciating between train and query, since train can contain additional transformations
        # for optimal training performance
        self.train_dataset = train_dataset
        self.query_dataset = query_dataset

        # Set up the indices for unlabeled and labeled pool
        self.unlabeled_indices = range(len(self.train_dataset))
        self.labeled_indices = []

    @property
    def unlabeled_dataset(self):
        return UnlabeledDataset(self.query_dataset, indices=self.unlabeled_indices)

    @property
    def labeled_dataset(self):
        return Subset(self.train_dataset, indices=self.labeled_indices)

    @property
    def full_dataset(self):
        # TODO: Is the train or query dataset required here?
        # For e.g., semi-supervised learning
        return FullDataset(self.train_dataset, unlabeled_indices=self.unlabeled_indices)

    def random_init(self, n_samples: int):
        """
            Randomly buys samples from the unlabeled pool and adds them to the labeled one.

            Args:
                n_samples (int): Size of the initial labeld pool.    
        """
        assert len(self.labeled_indices) == 0, 'Pools already initialized.'
        buy_idx = random.sample(self.unlabeled_indices, k=n_samples)
        self.labeled_indices = self.union(self.labeled_indices, buy_idx)
        self.unlabeled_indices = self.diff(self.unlabeled_indices, buy_idx)

    def update_annotations(self, buy_idx: list):
        """
            Updates the labeld pool with newly annotated samples.

            Args:
                buy_idx (list): List of indices which identify samples of the unlabeled pool that should be
                                transfered to the labeld pool.
        """
        self.labeled_indices = self.union(self.labeled_indices, buy_idx)
        self.unlabeled_indices = self.diff(self.unlabeled_indices, buy_idx)

    def union(self, a: list, b: list):
        return list(set(a).union(set(b)))

    def diff(self, a: list, b: list):
        return list(set(a).difference(set(b)))

    def __len__(self):
        return len(self.train_dataset)