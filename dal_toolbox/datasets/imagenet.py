import os
import warnings
import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets
from .base import AbstractData


class ImageNet(AbstractData):
    def __init__(
            self,
            dataset_path: str,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            val_split: float = 0.1,
            seed: int = None,
    ) -> None:
        self.mean = mean
        self.std = std
        path = self._check_path(dataset_path)
        super().__init__(path, val_split, seed)

    def _check_path(self, path):
        if os.path.basename(path) == 'ILSVRC2012':
            return path
        path = os.path.join(path, 'ILSVRC2012')
        return path

    @property
    def num_classes(self):
        return 1000

    def download_datasets(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"The path {self.dataset_path} does not exist.")

    @property
    def full_train_dataset(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.train_transforms)

    @property
    def full_train_dataset_eval_transforms(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.eval_transforms)

    @property
    def full_train_dataset_query_transforms(self):
        return datasets.ImageNet(self.dataset_path, split='train', transform=self.query_transforms)

    @property
    def test_dataset(self):
        return datasets.ImageNet(self.dataset_path, split='val', transform=self.eval_transforms)

    @property
    def train_transforms(self):
        train_transform = ClassificationPresetTrain(crop_size=224, mean=self.mean, std=self.std)
        return train_transform

    @property
    def eval_transforms(self):
        eval_transform = ClassificationPresetEval(crop_size=224, mean=self.mean, std=self.std)
        return eval_transform

    @property
    def query_transforms(self):
        return self.eval_transforms


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
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
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
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

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

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

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
