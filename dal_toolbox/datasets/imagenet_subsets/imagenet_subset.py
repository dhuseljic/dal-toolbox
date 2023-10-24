import os
from glob import glob

import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf

from dal_toolbox.datasets import ImageNet
from dal_toolbox.datasets.base import BaseTransforms
from dal_toolbox.datasets.imagenet import ImageNetContrastiveTransforms
from dal_toolbox.datasets.utils import PlainTransforms


class ImageNetSubSetWrapper(ImageNet):
    def __init__(self, dataset_path: str, transforms: BaseTransforms, val_split: float, seed: int,
                 subset_file: str, preload=False) -> None:
        self.subset_file = subset_file
        self.preload = preload
        super().__init__(dataset_path, transforms, val_split, seed)

    @property
    def full_train_dataset(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.train_transform, preload=self.preload)

    @property
    def full_train_dataset_eval_transforms(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.eval_transform, preload=self.preload)

    @property
    def full_train_dataset_query_transforms(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='train', transform=self.query_transform,
                              preload=False)  # Is not used currently, preloading not necessary

    @property
    def test_dataset(self):
        return ImageNetSubset(subset_file=self.subset_file, root=self.dataset_path,
                              split='val', transform=self.eval_transform, preload=self.preload)


class ImageNet50(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_50.txt", preload=preload)

    @property
    def num_classes(self):
        return 50


class ImageNet50Contrastive(ImageNet50):
    """
    Contrastive version of ImageNet50.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """

    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False, cds=1.0) -> None:
        super().__init__(dataset_path, ImageNetContrastiveTransforms(color_distortion_strength=cds), val_split, seed,
                         preload=preload)


class ImageNet50Plain(ImageNet50):
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, PlainTransforms(resize=(224, 224)), val_split, seed, preload=preload)


class ImageNet100(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_100.txt", preload=preload)

    @property
    def num_classes(self):
        return 100


class ImageNet100Plain(ImageNet100):
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, PlainTransforms(resize=(224, 224)), val_split, seed, preload=preload)


class ImageNet100Contrastive(ImageNet100):
    """
    Contrastive version of ImageNet100.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """

    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False, cds=1.0) -> None:
        super().__init__(dataset_path, ImageNetContrastiveTransforms(color_distortion_strength=cds), val_split, seed,
                         preload=preload)


class ImageNet200(ImageNetSubSetWrapper):
    def __init__(self, dataset_path: str, transforms: BaseTransforms = None, val_split: float = 0.1,
                 seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, transforms, val_split, seed, "imagenet_200.txt", preload=preload)

    @property
    def num_classes(self):
        return 200


class ImageNet200Contrastive(ImageNet200):
    """
    Contrastive version of ImageNet200.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """

    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False, cds=1.0) -> None:
        super().__init__(dataset_path, ImageNetContrastiveTransforms(color_distortion_strength=cds), val_split, seed,
                         preload=preload)


class ImageNet200Plain(ImageNet200):
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, preload=False) -> None:
        super().__init__(dataset_path, PlainTransforms(resize=(224, 224)), val_split, seed, preload=preload)


class ImageNetSubset(data.Dataset):
    # From https://github.com/avihu111/TypiClust/blob/main/scan/data/imagenet.py
    def __init__(self, subset_file, root, split='train', transform=None, preload=False):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, split)
        self.transform = transform
        self.split = split
        self.preload = preload

        # Read the subset of classes to include (sorted)
        with open(os.path.join(os.path.dirname(__file__), subset_file), 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)
        self.classes = class_names  # Contains list of class names

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                if self.preload:
                    with open(f, 'rb') as img_f:
                        img = Image.open(img_f).convert('RGB')
                    imgs.append((img, i))
                else:
                    imgs.append((f, i))
        self.imgs = imgs  # Contains (filepath, class) tuples

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, target = self.imgs[index]

        if not self.preload:
            with open(img, 'rb') as f:
                img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
