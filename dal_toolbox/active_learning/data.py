import random

import torch
import numpy as np
import lightning as L
from torch.utils.data import Subset, SubsetRandomSampler


class ALDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_ds,
            query_ds=None,
            train_batch_size: int = 32,
            predict_batch_size: int = 32,
            predict_type: str = 'unlabeled',
            seed: int = None,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.train_batch_size = train_batch_size

        if query_ds is None:
            print('Using train_dataset for queries. Make sure that there are no augmentations used.')
            query_ds = train_ds
        self.query_ds = query_ds
        self.predict_batch_size = predict_batch_size
        self.predict_type = predict_type

        # Set up the indices for unlabeled and labeled pool
        self.unlabeled_indices = range(len(self.train_ds))
        self.labeled_indices = []

        # set rng which should be used for all random stuff
        self._seed = seed
        if seed is None:
            self.rng = np.random.mtrand._rand
        else:
            self.rng = np.random.RandomState(self._seed)

    def train_dataloader(self):
        sampler = SubsetRandomSampler(indices=self.labeled_indices)
        train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.train_batch_size, sampler=sampler)
        return train_loader

    def predict_dataloader(self):
        if self.predict_type == 'unlabeled':
            sampler = self.unlabeled_indices
        elif self.predict_type == 'labeled':
            sampler = self.labeled_indices
        elif self.predict_type == 'all':
            sampler = range(len(self.query_ds))
        else:
            raise NotImplementedError(f'The predict type is not implemented, got {self.predict_type}')
        loader = torch.utils.data.DataLoader(self.query_ds, batch_size=self.train_batch_size, sampler=sampler)
        return loader

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()

    def update_annotations(self, buy_idx: list):
        """
            Updates the labeled pool with newly annotated instances.

            Args:
                buy_idx (list): List of indices which identify samples of the unlabeled pool that should be
                                transfered to the labeld pool.
        """
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
            classes = torch.Tensor([self.query_ds[idx][-1] for idx in self.unlabeled_indices]).long()
            classes_unique = classes.unique()
            n_classes = len(classes_unique)
            n_samples_per_class = n_samples // n_classes

            indices = []
            for label in classes_unique:
                unlabeled_indices_lbl = (classes == label).nonzero().squeeze()
                # indices_lbl = self.rng.sample(unlabeled_indices_lbl.tolist(), k=n_samples_per_class)
                indices_lbl = self.rng.choice(unlabeled_indices_lbl.tolist(), size=n_samples_per_class, replace=False)
                indices_lbl = indices_lbl.tolist()
                indices.extend(indices_lbl)
        else:
            # indices = self.rng.sample(self.unlabeled_indices, k=n_samples)
            indices = self.rng.choice(self.unlabeled_indices, size=n_samples, replace=False)
            indices = indices.tolist()

        self.update_annotations(indices)


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
    def __init__(self, train_dataset, query_dataset=None, random_state=None):
        # Differenciating between train and query, since train can contain additional transformations
        # for optimal training performance
        self.train_dataset = train_dataset
        if query_dataset is None:
            print('Using train_dataset for queries. Make sure that there are no augmentations used.')
            query_dataset = train_dataset
        self.query_dataset = query_dataset

        # Set up the indices for unlabeled and labeled pool
        self.unlabeled_indices = range(len(self.train_dataset))
        self.labeled_indices = []
        self.rng = random.Random(random_state)

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

    def update_annotations(self, buy_idx: list):
        """
            Updates the labeled pool with newly annotated samples.

            Args:
                buy_idx (list): List of indices which identify samples of the unlabeled pool that should be
                                transfered to the labeld pool.
        """
        self.labeled_indices = list_union(self.labeled_indices, buy_idx)
        self.unlabeled_indices = list_diff(self.unlabeled_indices, buy_idx)

    def __len__(self):
        return len(self.train_dataset)

    def state_dict(self):
        return {'unlabeled_indices': self.unlabeled_indices, 'labeled_indices': self.labeled_indices}

    def load_state_dict(self, state_dict):
        necessary_keys = self.state_dict().keys()

        # Check for wrong keys
        for key in state_dict.keys():
            if key not in necessary_keys:
                raise ValueError(f'The key `{key}` can not be loaded.')
        # Notify that some keys are not loaded
        for key in necessary_keys:
            if key not in state_dict.keys():
                print(f'<Key `{key}` is not present and not loaded>')

        for key in state_dict:
            setattr(self, key, state_dict[key])
        print('<All keys matched successfully>')

    def random_init(self, n_samples: int, class_balanced: bool = False):
        """Randomly buys samples from the unlabeled pool and adds them to the labeled one.

            Args:
                n_samples (int): Size of the initial labeld pool.    
                class_balanced (bool): Whether to use an class balanced initialization.
        """
        if len(self.labeled_indices) != 0:
            raise ValueError('Pools already initialized.')

        if class_balanced:
            classes = torch.Tensor([self.query_dataset[idx][-1] for idx in self.unlabeled_indices]).long()
            classes_unique = classes.unique()
            n_classes = len(classes_unique)
            n_samples_per_class = n_samples // n_classes

            indices = []
            for label in classes_unique:
                unlabeled_indices_lbl = (classes == label).nonzero().squeeze()
                indices_lbl = self.rng.sample(unlabeled_indices_lbl.tolist(), k=n_samples_per_class)
                indices.extend(indices_lbl)
        else:
            indices = self.rng.sample(self.unlabeled_indices, k=n_samples)

        self.update_annotations(indices)


def list_union(a: list, b: list):
    return list(set(a).union(set(b)))


def list_diff(a: list, b: list):
    return list(set(a).difference(set(b)))
