import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dal_toolbox.models.utils.base import BaseModule


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


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
        features, labels = model.get_representations(dataloader, device, return_labels=True)
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
