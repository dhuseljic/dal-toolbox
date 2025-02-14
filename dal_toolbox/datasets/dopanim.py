import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.preprocessing import LabelEncoder
from enum import Enum

from .base import BaseData, BaseTransforms

# class DopanimTransforms(Enum):
#     mean: tuple = (0.485, 0.456, 0.406)
#     std: tuple = (0.229, 0.224, 0.225)


class Dopanim(BaseData):

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
        return 15

    def download_datasets(self):
        pass
        # _Dopanim(self.dataset_path, version="train", download=True)
        # _Dopanim(self.dataset_path, version="test", download=True)

    @property
    def full_train_dataset(self):
        return _Dopanim(self.dataset_path, version="train", transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return _Dopanim(self.dataset_path, version="train", transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return _Dopanim(self.dataset_path, version="train", transform=self.query_transform)

    @property
    def test_dataset(self):
        return _Dopanim(self.dataset_path, version="test", transform=self.eval_transform)


class _Dopanim(Dataset):
    """Dopanim

    The dopanim [1, 2] dataset features about 15,750 animal images of 15 classes, organized into four groups of
    doppelganger animals and collected together with ground truth labels from iNaturalist. For approximately 10,500 of
    these images, 20 humans provided over 52,000 annotations with an accuracy of circa 67%.

    Parameters
    ----------
    root : str
        Path to the root directory, where the ata is located.
    version : "train" or "valid" or "test", default="train"
        Defines the version (split) of the dataset.
    download : bool, default=False
        Flag whether the dataset will be downloaded.
    transforms : "auto" or torch.nn.Module, default="auto"
        Transforms for the samples, where "auto" used pre-defined transforms fitting the respective version.

    References
    ----------
    [1] Herde, M., Huseljic, D., Rauch, L., & Sick, B. (2024) dopanim: A Dataset of Doppelganger Animals with Noisy
        Annotations from Multiple Humans. In NeurIPS: D & B Track.
    [2] Herde, M., Huseljic, D., Rauch, L., & Sick, B. (2024). dopanim: A Dataset of Doppelganger Animals with Noisy
        Annotations from Multiple Humans [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11479589
    """

    base_folder = "dopanim_14016659"
    url = "https://zenodo.org/api/records/14016659/files-archive"
    filename = "14016659.zip"
    image_dir = "images"
    classes = np.array(
        [
            "German Yellowjacket",
            "European Paper Wasp",
            "Yellow-legged Hornet",
            "European Hornet",
            "Brown Hare",
            "Black-tailed Jackrabbit",
            "Marsh Rabbit",
            "Desert Cottontail",
            "European Rabbit",
            "Eurasian Red Squirrel",
            "American Red Squirrel",
            "Douglas' Squirrel",
            "Cheetah",
            "Jaguar",
            "Leopard",
        ],
        dtype=object,
    )
    def __init__(
        self,
        root: str,
        version: str = "train",
        download: bool = False,
        transform=None,
    ):
        # Download data.
        self.folder = os.path.join(root, _Dopanim.base_folder)
        if download:
            download_and_extract_archive(
                _Dopanim.url, root, filename=_Dopanim.filename, extract_root=self.folder)
            version_filename = os.path.join(self.folder, f"{version}.zip")
            download_and_extract_archive(
                _Dopanim.url, root, filename=version_filename, extract_root=self.folder)

        # Check availability of data.
        is_available = os.path.exists(self.folder)
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load annotation file.
        if version not in ["train", "valid", "test"]:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.img_folder = os.path.join(self.folder, version)

        # Load and prepare true labels as tensor.
        self.y_orig, self.observation_ids = self.load_true_class_labels(version=version)
        self.le = LabelEncoder().fit(self.y_orig)
        self.y = self.le.transform(self.y_orig)
        self.transform = transform

        # Transform to tensors.
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        """
        Returns
        -------
        length: int
            Length of the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        sample : torch.tensor
            Sample with the given index.
        """
        x = Image.open(os.path.join(self.img_folder, self.y_orig[idx], f"{self.observation_ids[idx]}.jpeg"))
        x = x.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        y = self.y[idx]
        return x, y

    def load_true_class_labels(self, version="train"):
        """
        Loads the true class of the given version.

        Parameters
        ----------
        version : "train" or "valid" or "test"
            Version (split) of the dataset.

        Returns
        -------
        z : np.ndarray of shape (n_samples,)
            True class labels.
        """
        with open(os.path.join(self.folder, "task_data.json")) as task_file:
            task_data = json.load(task_file)
        y_true_list = []
        observation_id_list = []
        for observation_id, observation_dict in task_data.items():
            if observation_dict["split"] == version:
                y_true_list.append(observation_dict["taxon_name"])
                observation_id_list.append(observation_id)
        return np.array(y_true_list, dtype=object), np.array(observation_id_list, dtype=int)
