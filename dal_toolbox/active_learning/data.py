
import torch
import numpy as np
import lightning as L

from torch.utils.data import Subset, RandomSampler, DataLoader, Dataset

from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from ..utils import setup_rng


class ActiveLearningDataModule(L.LightningDataModule):
    # TODO(dhuseljic): Implement for LightningDataModule input.
    def __init__(
            self,
            train_dataset: Dataset,
            query_dataset: Dataset = None,
            val_dataset: Dataset = None,
            test_dataset: Dataset = None,
            train_batch_size: int = 64,
            predict_batch_size: int = 256,
            seed: int = None,
            collator = None
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.query_dataset = QueryDataset(query_dataset) if query_dataset else QueryDataset(dataset=train_dataset)
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.collator = collator

        if query_dataset is None:
            rank_zero_warn('Using train_dataset for queries. Ensure that there are no augmentations used.')

        if self.val_dataset is not None:
            self.val_dataloader = lambda: DataLoader(self.val_dataset, batch_size=predict_batch_size, shuffle=False)

        if self.test_dataset is not None:
            self.test_dataloader = lambda: DataLoader(self.test_dataset, batch_size=predict_batch_size, shuffle=False)

        self.rng = setup_rng(seed)
        self.unlabeled_indices = list(range(len(self.train_dataset)))
        self.labeled_indices = []

    def train_dataloader(self):
        # TODO(dhuseljic): Add support for semi-supervised learning loaders.
        labeled_dataset = Subset(self.train_dataset, indices=self.labeled_indices)
        iter_per_epoch = len(labeled_dataset) // self.train_batch_size + 1
        sampler = RandomSampler(labeled_dataset, num_samples=(iter_per_epoch * self.train_batch_size))
        train_loader = DataLoader(labeled_dataset, batch_size=self.train_batch_size, sampler=sampler, collate_fn=self.collator)
        return train_loader

    def unlabeled_dataloader(self, subset_size=None):
        """Returns a dataloader for the unlabeled pool where instances are not augmentated."""
        unlabeled_indices = self.unlabeled_indices
        if subset_size is not None:
            unlabeled_indices = self.rng.choice(unlabeled_indices, size=subset_size, replace=False)
            unlabeled_indices = unlabeled_indices.tolist()
        loader = DataLoader(self.query_dataset, batch_size=self.predict_batch_size, sampler=unlabeled_indices, collate_fn=self.collator)
        return loader, unlabeled_indices

    def labeled_dataloader(self, subset_size=None):
        """Returns a dataloader for the labeled pool where instances are not augmentated."""
        labeled_indices = self.labeled_indices
        if subset_size is not None:
            labeled_indices = self.rng.choice(labeled_indices, size=subset_size, replace=False)
            labeled_indices = labeled_indices.tolist()
        loader = DataLoader(self.query_dataset, batch_size=self.predict_batch_size, sampler=labeled_indices, collate_fn=self.collator)
        return loader, labeled_indices

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
                indices_lbl = self.rng.choice(unlabeled_indices_lbl.tolist(), size=n_samples_per_class, replace=False)
                indices_lbl = indices_lbl.tolist()
                indices.extend(indices_lbl)
        else:
            indices = self.rng.choice(self.unlabeled_indices, size=n_samples, replace=False)
            indices = indices.tolist()
        self.update_annotations(indices)


class QueryDataset(Subset):
    """A helper class which returns also the index along with the instances and targets."""
    #problem with dictionary output of dataset

    def __init__(self, dataset):
        super().__init__(dataset=dataset, indices=range(len(dataset)))
        self.dataset = dataset

    def __getitem__(self, index):
        # TODO(dhuseljic): discuss with marek, index instead of target? maybe dictionary? leave it like that?
        data = super().__getitem__(index)
        if isinstance(data, dict):
            instance = {
                'input_ids': data['input_ids'], 
                'attention_mask': data['attention_mask'], 
                'label': data['label'],
                'index': index
            }
            return instance
        else:
            instance, target = super().__getitem__(index)
            return instance, target, index


class ALDataset:
    # TODO: Update?
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
        self._setup_rng(random_state)

    def _setup_rng(self, seed):
        # set rng which should be used for all random stuff
        self._seed = seed
        if seed is None:
            self.rng = np.random.mtrand._rand
        else:
            self.rng = np.random.RandomState(self._seed)

    @property
    def unlabeled_dataset(self):
        return Subset(self.query_dataset, indices=self.unlabeled_indices)

    @property
    def labeled_dataset(self):
        return Subset(self.train_dataset, indices=self.labeled_indices)

    def update_annotations(self, buy_idx: list):
        """
            Updates the labeled pool with newly annotated samples.

            Args:
                buy_idx (list): List of indices which identify samples of the unlabeled pool that should be
                                transfered to the labeled pool.
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


class ALModule(ActiveLearningDataModule):
    # TODO(dhuseljic): How ot get query dataset from datamodule of lightning
    def __init__(self, datamodule: L.LightningDataModule, predict_batch_size: int = 256, seed: int = None):
        # Get datasets
        datamodule.prepare_data()
        datamodule.setup('fit')
        datamodule.setup('validate')
        datamodule.setup('test')
        train_loader = datamodule.train_dataloader()

        train_dataset = train_loader.dataset
        train_batch_size = train_loader.batch_size

        try:
            val_dataloader = datamodule.val_dataloader()
            val_dataset = val_dataloader.dataset
        except MisconfigurationException:
            print('Did not find validation dataloader. Using none.')
            val_dataset = None

        try:
            test_dataloader = datamodule.test_dataloader()
            test_dataset = test_dataloader.dataset
        except MisconfigurationException:
            print('Did not find test dataloader. Using none.')
            test_dataset = None
        super().__init__(train_dataset, train_dataset, val_dataset, test_dataset, train_batch_size, predict_batch_size, seed=seed)


def list_union(a: list, b: list):
    return list(set(a).union(set(b)))


def list_diff(a: list, b: list):
    return list(set(a).difference(set(b)))
