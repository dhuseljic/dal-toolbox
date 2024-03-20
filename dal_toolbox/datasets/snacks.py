import numpy as np
from enum import Enum

from .base import BaseData, BaseTransforms
from .utils import PlainTransforms
from datasets import load_dataset, config
from torch.utils.data import Dataset


class SnacksTransforms(Enum):
    # TODO
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class Snacks(BaseData):

    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        self.transforms = PlainTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 20

    def download_datasets(self):
        _Snacks(self.dataset_path, split="train")
        _Snacks(self.dataset_path, split="test")

    @property
    def full_train_dataset(self):
        return _Snacks(self.dataset_path, split="train", transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return _Snacks(self.dataset_path, split="train", transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return _Snacks(self.dataset_path, split="train", transform=self.query_transform)

    @property
    def test_dataset(self):
        return _Snacks(self.dataset_path, split="test", transform=self.eval_transform)


class _Snacks(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        split_mapping = {'train': 'train', 'test': 'test'}

        config.DOWNLOADED_DATASETS_PATH = root
        config.HF_DATASETS_CACHE = root
        self.ds = load_dataset("Matthijs/snacks", split=split_mapping[self.split])
        # self.ds_dict = self.ds.train_test_split(test_size=0.1, seed=42)
        # self.ds = self.ds_dict[self.split]
        # self.target_dict = {k: i for i, k in enumerate(np.unique(self.ds['target']))}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        data = self.ds[index]
        img = data['image']
        lbl = data['label']

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl
