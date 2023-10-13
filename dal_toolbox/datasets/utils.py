import warnings

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from dal_toolbox.datasets.base import BaseTransforms
from dal_toolbox.datasets.base import BaseData
from dal_toolbox.models.utils.base import BaseModule


class RepeatTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class PlainTransforms(BaseTransforms):
    def __init__(self, resize=None):
        if resize:
            self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize), torchvision.transforms.ToTensor()])
        else:
            self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class FeatureDatasetWrapper(BaseData):
    """
    Wrapper for FeatureDatasets to be used with AbstractData
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    @property
    def num_classes(self):
        return self.n_classes

    @property
    def num_features(self):
        return self.n_features

    def download_datasets(self):
        map = "cpu" if not torch.cuda.is_available() else None
        feature_dict = torch.load(self.dataset_path, map_location=map)
        self._trainset = feature_dict["trainset"]
        self._testset = feature_dict["testset"]
        self.n_classes = len(torch.unique(self._trainset.labels))
        self.n_features = self._trainset.features.shape[1]

    @property
    def full_train_dataset_eval_transforms(self):
        warnings.warn("FeatureDataset hast no EvalTransforms")
        return self.full_train_dataset

    @property
    def full_train_dataset_query_transforms(self):
        warnings.warn("FeatureDataset hast no QueryTransform")
        return self.full_train_dataset

    @property
    def test_dataset(self):
        return self._testset

    @property
    def train_transforms(self):
        return None

    @property
    def query_transforms(self):
        return None

    @property
    def eval_transforms(self):
        return None

    @property
    def full_train_dataset(self):
        return self._trainset


class FeatureDataset(Dataset):
    """
    Dataset for feature representations of a model.

    This dataset class takes a ``model`` and a ``dataset`` and saves the features to use for later. Some tasks (e.g. the
    linear evaluation accuracy) need datasets that entail the feature representations of a model.
    """

    def __init__(self, model: BaseModule, dataset: Dataset, device: torch.device) -> None:
        """
        Initializes ``FeatureDataset``.
        Args:
            model: The model the features are extracted from.
            dataset: The dataset from which the features are extracted.
            device: The ``torch.device``, with which the features are extracted
        """
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)
        features, labels = model.get_representations(dataloader, device=device, return_labels=True)
        self.features = features.detach()
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return self.features[idx], self.labels[idx]


def sample_balanced_subset(targets, num_samples):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # Get samples per class
    num_classes = len(torch.unique(targets))
    assert num_samples % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
    num_samples_per_class = [int(num_samples / num_classes)] * num_classes

    val_pool = []
    for c in range(num_classes):
        idx = np.array([i for i in range(len(targets)) if targets[i] == c])
        np.random.shuffle(idx)
        val_pool.extend(idx[:num_samples_per_class[c]])
    return [int(i) for i in val_pool]
