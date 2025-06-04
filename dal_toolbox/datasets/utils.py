import os
import hashlib
import warnings

import numpy as np
import torch

import torchvision
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor

from dal_toolbox.datasets.base import BaseTransforms
from dal_toolbox.datasets.base import BaseData
from dal_toolbox.models.utils.base import BaseModule

import torch


class ViTMAETransforms():
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

        self.transform = torchvision.transforms.Compose([
            # First create three channels if black and white
            torchvision.transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            self.image_processor                  # then apply processor
        ])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class SwinV2Transforms():
    def __init__(self, backbone):
        if backbone == 'swinv2':
            self.image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/swinv2-base-patch4-window8-256")
        elif backbone == 'swinv2-t':
            self.image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/swinv2-tiny-patch4-window8-256")
        elif backbone == 'swinv2-s':
            self.image_processor = AutoImageProcessor.from_pretrained(
                "microsoft/swinv2-small-patch4-window8-256")
        else:
            raise AssertionError("Wrong backbone!")

        self.transform = torchvision.transforms.Compose([
            # First create three channels if black and white
            torchvision.transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            self.image_processor                  # then apply processor
        ])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class DinoTransforms():
    def __init__(self, size=None, center_crop_size=224):
        if size:
            # https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L65-L70
            dino_mean = (0.485, 0.456, 0.406)
            dino_std = (0.229, 0.224, 0.225)
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size, interpolation=3),
                torchvision.transforms.CenterCrop(center_crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) != 3 else x),
                torchvision.transforms.Normalize(dino_mean, dino_std),
            ])

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


class FeatureDataset(Dataset):
    """A PyTorch Dataset that extracts and/or caches features from a given model and dataset."""

    def __init__(self, model, dataset, cache=False, cache_dir=None, batch_size=256,  num_workers=8, device='cuda'):
        """
        Args:
            model (nn.Module): A PyTorch model used to extract features.
            dataset (Dataset): A Dataset whose __getitem__ returns either:
                               - For vision: (image_tensor, label)
                               - For text: {'input_ids': ..., 'attention_mask': ..., 'label': ...}
            cache (bool): If True, save/load features to/from disk.
            cache_dir (str, optional): Directory where cached features are stored. If None, defaults to ~/.cache/feature_datasets.
            batch_size (int): Batch size for feature extraction.
            device (str): Device on which to run feature extraction ('cuda' or 'cpu').
            task (str, optional): Either "text" (for models expecting input_ids+attention_mask) or None (default for vision).
        """
        if cache:
            if cache_dir is None:
                home_dir = os.path.expanduser('~')
                cache_dir = os.path.join(home_dir, '.cache', 'feature_datasets')
            os.makedirs(cache_dir, exist_ok=True)

            hash = self._create_hash(dataset, model)
            file_name = os.path.join(cache_dir, hash + '.pth')

            if os.path.exists(file_name):
                print('Loading cached features from', file_name)
                features, labels = torch.load(file_name, map_location='cpu')
            else:
                features, labels = self._extract_features(
                    model=model,
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device
                )
                print('Saving features to cache file', file_name)
                torch.save((features, labels), file_name)
        else:
            features, labels = self._extract_features(model, dataset, batch_size, device)

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    def _create_hash(self, dataset, model, num_hash_samples=50):
        """Creates an MD5 hash based on:
          - Number of samples in the dataset
          - Model's total number of parameters
          - A small subset of samples (to detect dataset changes)

        This helps to invalidate the cache whenever the dataset or model changes.
        """
        hasher = hashlib.md5()

        num_samples = len(dataset)
        hasher.update(str(num_samples).encode())

        num_parameters = sum([p.numel() for p in model.parameters()])
        hasher.update(str(model).encode())
        hasher.update(str(num_parameters).encode())

        indices_to_hash = range(0, num_samples, num_samples//num_hash_samples)
        for idx in indices_to_hash:
            sample = dataset[idx][0]
            hasher.update(str(sample).encode())
        return hasher.hexdigest()

    @torch.no_grad()
    def _extract_features(self, model, dataset, batch_size, num_workers, device):
        print('Extracting features from model...')
        model.eval()
        model.to(device)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        features = []
        labels = []
        for batch in dataloader:
            features.append(model(batch[0].to(device)).to('cpu'))
            labels.append(batch[1])
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels
