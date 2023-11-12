import os
from glob import glob

import torch.utils.data as data
from PIL import Image

from dal_toolbox.datasets import ImageNet
from dal_toolbox.datasets.base import BaseTransforms
from dal_toolbox.datasets.imagenet import ImageNetContrastiveTransforms
from dal_toolbox.datasets.utils import PlainTransforms


class ImageNetSubSetWrapper(ImageNet):
    def __init__(self, dataset_path: str, transforms: BaseTransforms, val_split: float, seed: int,
                 subset_file: str, preload=False) -> None:
        self.subset_file = subset_file
        self.dataset_path = dataset_path

        self.trainset = ImageNetLoader(subset_file=subset_file, root=dataset_path, split='train', preload=preload)
        self.testset = ImageNetLoader(subset_file=subset_file, root=dataset_path, split='val', preload=preload)
        super().__init__(dataset_path, transforms, val_split, seed)

    @property
    def full_train_dataset(self):
        return ImageNetSubset(imgs=self.trainset.imgs, class_names=self.trainset.classes, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return ImageNetSubset(imgs=self.trainset.imgs, class_names=self.trainset.classes, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return ImageNetSubset(imgs=self.trainset.imgs, class_names=self.trainset.classes, transform=self.query_transform)

    @property
    def test_dataset(self):
        return ImageNetSubset(imgs=self.testset.imgs, class_names=self.testset.classes, transform=self.eval_transform)


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


class ImageNetLoader:
    # From https://github.com/avihu111/TypiClust/blob/main/scan/data/imagenet.py
    def __init__(self, subset_file, root, split='train', preload=False):
        self.root = os.path.join(root, split)
        self.split = split

        print(f"Loading {split} split of ImageNet {subset_file}")
        
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
                if preload:
                    with open(f, 'rb') as img_f:
                        img = Image.open(img_f).convert('RGB')
                    imgs.append((img, i))
                else:
                    imgs.append((f, i))
        self.imgs = imgs  # Contains (filepath, class) tuples


class ImageNetSubset(data.Dataset):
    def __init__(self, imgs, class_names, transform=None):
        super(ImageNetSubset, self).__init__()
        self.imgs = imgs
        self.class_names = class_names
        self.transform = transform

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, target = self.imgs[index]

        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
