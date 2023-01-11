import json
import random

import torch
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
    def __init__(self, train_dataset, query_dataset=None):
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
        self.rng = random.Random(0)

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

    def random_init(self, n_samples: int):
        """Randomly buys samples from the unlabeled pool and adds them to the labeled one.

            Args:
                n_samples (int): Size of the initial labeld pool.    
        """
        if len(self.labeled_indices) != 0:
            raise ValueError('Pools already initialized.')
        indices = self.rng.sample(self.unlabeled_indices, k=n_samples)
        self.labeled_indices = list_union(self.labeled_indices, indices)
        self.unlabeled_indices = list_diff(self.unlabeled_indices, indices)

    def load_init(self, result_json: str):
        """Initializes the unlabeled pool using the one used in an already run experiment.

            Args:
                result_json (str): The result json file from the already run experiment.
        """
        with open(result_json, 'r') as f:
            loaded_results = json.load(f)
        cycle_results = loaded_results['cycle0']
        indices = cycle_results['labeled_indices']
        self.labeled_indices = list_union(self.labeled_indices, indices)
        self.unlabeled_indices = list_diff(self.unlabeled_indices, indices)




def list_union(a: list, b: list):
    return list(set(a).union(set(b)))


def list_diff(a: list, b: list):
    return list(set(a).difference(set(b)))
