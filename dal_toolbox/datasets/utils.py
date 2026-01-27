import os
import hashlib

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class FeatureDataset(Dataset):
    """A PyTorch Dataset that extracts and/or caches features from a given model and dataset."""

    def __init__(self, model, dataset, cache=False, cache_dir=None, batch_size=256, num_workers=8, pbar=True, device='cuda'):
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
        self.pbar = tqdm if pbar else (lambda x, **kwargs: x)
        if cache:
            if cache_dir is None:
                home_dir = os.path.expanduser('~')
                cache_dir = os.path.join(home_dir, '.cache', 'feature_datasets')
            cache_dir = os.path.join(cache_dir, 'feature_datasets')
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
            features, labels = self._extract_features(model, dataset, batch_size, num_workers, device)

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def _create_hash(self, dataset, model, num_hash_samples=50):
        """Creates an MD5 hash based on:
          - Number of samples in the dataset
          - Model's total number of parameters
          - A small subset of samples (to detect dataset changes)

        This helps to invalidate the cache whenever the dataset or model changes.
        """
        model.cpu()
        hasher = hashlib.md5()

        num_samples = len(dataset)
        hasher.update(str(num_samples).encode())

        num_parameters = sum([p.numel() for p in model.parameters()])
        hasher.update(str(num_parameters).encode())

        indices_to_hash = range(0, num_samples, num_samples//num_hash_samples)
        for idx in indices_to_hash:
            sample = dataset[idx][0]
            hasher.update(sample.numpy().tobytes())

        for _, param in model.state_dict().items():
            hasher.update(param.cpu().numpy().tobytes())
        return hasher.hexdigest()

    @torch.no_grad()
    def _extract_features(self, model, dataset, batch_size, num_workers, device):
        model.eval()
        model.to(device)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        features = []
        labels = []
        for batch in self.pbar(dataloader, desc='Extracting features'):
            features.append(model(batch[0].to(device)).to('cpu'))
            labels.append(batch[1])
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels


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


class LongTailedWrapper(Dataset):
    """A wrapper that makes any dataset long-tailed by subsampling it."""

    def __init__(self, dataset, imbalance_ratio, imb_type='exp', seed=42):
        self.dataset = dataset
        self.imbalance_ratio = imbalance_ratio
        self.imb_type = imb_type
        self.seed = seed

        # 1. Attempt to extract targets/labels from the underlying dataset
        # Most standard datasets (CIFAR, MNIST, ImageFolder) use .targets
        if hasattr(dataset, 'targets'):
            self.targets = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.targets = np.array(dataset.labels)
        else:
            raise AttributeError("Dataset must have .targets or .labels attribute to determine classes.")

        self.classes = np.unique(self.targets)
        self.num_classes = len(self.classes)

        # 2. Calculate how many images we need per class
        self.img_num_list = self._get_img_num_per_cls(
            len(self.dataset), self.num_classes, imb_type, imbalance_ratio)

        # 3. Generate the list of valid indices (the "Long Tail" mask)
        self.indices = self._gen_imbalanced_indices()

        # 4. Map the new (reduced) targets for easy access
        self.imbalanced_targets = torch.from_numpy(self.targets[self.indices])

    def _get_img_num_per_cls(self, total_data_len, cls_num, imb_type, imb_factor):
        """Calculates the number of samples needed for each class (0 to N-1)."""
        # We assume the original dataset is balanced or roughly balanced for this calculation
        img_max = total_data_len / cls_num
        img_num_per_cls = []

        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                # imb_factor = max / min
                num = img_max * (1.0 / imb_factor) ** (cls_idx / (cls_num - 1.0))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * (1.0 / imb_factor)))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        return img_num_per_cls

    def _gen_imbalanced_indices(self):
        """Selects indices from the original dataset to match the desired distribution. """
        np.random.seed(self.seed)
        all_indices = []

        # Map class ID to how many samples we want
        # Note: We assume classes are sorted 0..N for the decay (0=Head, N=Tail)
        for i, the_class in enumerate(self.classes):
            target_num = self.img_num_list[i]

            # Find all original indices for this class
            # We use np.where on the original full targets
            indices_for_class = np.where(self.targets == the_class)[0]

            # Shuffle and pick top N
            np.random.shuffle(indices_for_class)
            selected_indices = indices_for_class[:target_num]
            all_indices.extend(selected_indices)

        return all_indices

    def __getitem__(self, index):
        # Map the requested index to the actual index in the original dataset
        real_index = self.indices[index]
        return self.dataset[real_index]

    def __len__(self):
        return len(self.indices)
