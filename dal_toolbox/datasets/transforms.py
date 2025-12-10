import torch
from torchvision.transforms import Compose, Resize, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize
from torchvision.transforms import autoaugment, PILToTensor, ConvertImageDtype, CenterCrop, RandomResizedCrop
from torchvision.transforms import RandomErasing
from torchvision.transforms.functional import InterpolationMode
from .base import BaseTransforms

# TODO: Add default Transforms that is used for Cifar, ImageNet, etc.


class CustomTransforms(BaseTransforms):
    """
    A flexible wrapper class for custom image transformations.

    This class allows the user to specify separate transformation pipelines for 
    training, evaluation, and optionally querying. If no query transform is provided, 
    the evaluation transform is used as a default for querying.

    Args:
        train_transform (callable): Transformation applied during training.
        eval_transform (callable): Transformation applied during evaluation.
        query_transform (callable, optional): Transformation applied during querying.
                                              Defaults to `eval_transform` if not specified.
    """

    def __init__(self, train_transform, eval_transform, query_transform=None):
        super().__init__()
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        self._query_transform = eval_transform if query_transform is None else query_transform

    @property
    def train_transform(self):
        return self._train_transform

    @property
    def query_transform(self):
        return self._query_transform

    @property
    def eval_transform(self):
        return self._eval_transform


class PlainTransforms(BaseTransforms):
    """
    A simple transformation pipeline for image data.

    This class applies minimal preprocessing to the input images.
    If a `resize` argument is provided, images will be resized to the given size
    before being converted to tensors. Otherwise, only tensor conversion is applied.

    Args:
        resize (tuple or None): Target size as (height, width) to resize the image.
                                If None, resizing is skipped.

    Attributes:
        transform (Compose): Composed transformation pipeline.
    """

    def __init__(self, resize=None):
        if resize:
            self.transform = Compose([Resize(resize), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class StandardTransforms(BaseTransforms):
    """
    A standard transformation pipeline commonly used for CIFAR-10 or CIFAR-100.

    This class applies commonly used image transformations for training and evaluation
    of image classification models. It includes random cropping and flipping for data 
    augmentation during training, and normalization using dataset-specific statistics.

    Args:
        mean (list or tuple): Mean values for each channel used in normalization.
        std (list or tuple): Standard deviation values for each channel used in normalization.

    Attributes:
        mean (list): Mean for normalization.
        std (list): Standard deviation for normalization.
    """

    def __init__(self, resize, mean, std):
        super().__init__()
        self.resize = resize
        self.mean = mean
        self.std = std

    @property
    def train_transform(self):
        transform = Compose([
            RandomCrop(self.resize, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def eval_transform(self):
        return Compose([ToTensor(), Normalize(self.mean, self.std)])

    @property
    def query_transform(self):
        return Compose([ToTensor(), Normalize(self.mean, self.std)])


class MultiTransform:  # TODO to utils
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, x):
        return [transform(x) for transform in self.transforms]


# Transforms from: https://github.com/pytorch/vision/blob/main/references/classification/presets.py
class TransformPresetTrain:
    def __init__(
            self,
            *,
            crop_size,
            mean,
            std,
            interpolation=InterpolationMode.BILINEAR,
            hflip_prob=0.5,
            auto_augment_policy=None,
            ra_magnitude=9,
            augmix_severity=3,
            random_erase_prob=0.0,
    ):
        trans = [RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
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
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(RandomErasing(p=random_erase_prob))

        self.transforms = Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class TransformPresetEval:
    def __init__(
            self,
            *,
            crop_size,
            resize_size=256,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
    ):
        self.transforms = Compose([
            Resize(resize_size, interpolation=interpolation),
            CenterCrop(crop_size),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)
