import os
from glob import glob

import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf

from dal_toolbox.datasets import ImageNet
from dal_toolbox.datasets.base import BaseTransforms


class ImageNetSubSetWrapper(ImageNet):
    def __init__(self, dataset_path: str, transforms: BaseTransforms, val_split: float, seed: int,
                 subset_file: str) -> None:
        self.subset_file = subset_file
        super().__init__(dataset_path, transforms, val_split, seed)

    @property
    def full_train_dataset(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.query_transform)

    @property
    def test_dataset(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='val', transform=self.eval_transform)


class ImageNet50(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_50.txt")

    @property
    def num_classes(self):
        return 50


class ImageNet100(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_100.txt")

    @property
    def num_classes(self):
        return 100


class ImageNet200(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_200.txt")

    @property
    def num_classes(self):
        return 200


class ImageNetSubset(data.Dataset):
    # From https://github.com/avihu111/TypiClust/blob/main/scan/data/imagenet.py
    def __init__(self, subset_file, root, split='train', transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, split)
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(os.path.join(os.path.dirname(__file__), subset_file), 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs  # Contains (filepath, class) tuples
        self.classes = class_names  # Contains list of class names

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
