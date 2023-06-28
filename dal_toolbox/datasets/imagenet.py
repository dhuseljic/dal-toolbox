import os
import warnings
import torch
import torchvision
from enum import Enum
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets
from .base import BaseData, BaseTransforms


class ImageNetTransforms(Enum):
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class ImageNetStandardTransforms(BaseTransforms):
    def __init__(self) -> None:
        super().__init__()
        self.mean = ImageNetTransforms.mean.value
        self.std = ImageNetTransforms.std.value

    @property
    def train_transform(self):
        transform = ClassificationPresetTrain(crop_size=224, mean=self.mean, std=self.std)
        return transform

    @property
    def eval_transform(self):
        transform = ClassificationPresetEval(crop_size=224, mean=self.mean, std=self.std)
        return transform

    @property
    def query_transform(self):
        transform = ClassificationPresetEval(crop_size=224, mean=self.mean, std=self.std)
        return transform


class ImageNet(BaseData):
    def __init__(
            self,
            dataset_path: str,
            transforms: BaseTransforms = None,
            val_split: float = 0.1,
            seed: int = None,
    ) -> None:
        path = self._check_path(dataset_path)
        transforms = ImageNetStandardTransforms() if transforms is None else transforms
        self.train_transform = transforms.train_transform
        self.eval_transform = transforms.eval_transform
        self.query_transform = transforms.query_transform
        super().__init__(path, val_split, seed)

    def _check_path(self, path):
        if os.path.basename(path) == 'ILSVRC2012':
            return path
        if 'ILSVRC2012' in os.listdir(path):
            path = os.path.join(path, 'ILSVRC2012')
            return path
        if 'imagenet' in os.listdir(path):
            path = os.path.join(path, 'imagenet', 'ILSVRC2012')
            return path
        return path

    @property
    def num_classes(self):
        return 1000

    def download_datasets(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"The path {self.dataset_path} does not exist.")

    @property
    def full_train_dataset(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.query_transform)

    @property
    def test_dataset(self):
        return datasets.ImageNet(self.dataset_path, split='val', transform=self.eval_transform)


# Transforms from: https://github.com/pytorch/vision/blob/main/references/classification/presets.py
class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [torchvision.transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(torchvision.transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(torchvision.transforms.RandomErasing(p=random_erase_prob))

        self.transforms = torchvision.transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_size, interpolation=interpolation),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)


def build_imagenet(split, ds_path, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), return_info=False):
    warnings.warn('Deprecated method build_imagenet.')
    train_transform = ClassificationPresetTrain(crop_size=32, mean=mean, std=std)
    eval_transform = ClassificationPresetEval(crop_size=32, mean=mean, std=std)
    if split == 'train':
        ds = datasets.ImageNet(root=ds_path, split=split, transform=train_transform)
    elif split == 'query':
        ds = datasets.ImageNet(root=ds_path, split=split, transform=eval_transform)
    elif split == 'val':
        ds = datasets.ImageNet(root=ds_path, split=split, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 1000, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
