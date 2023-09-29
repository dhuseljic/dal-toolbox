import warnings

import torchvision
from enum import Enum
from .base import BaseData, BaseTransforms
from .utils import PlainTransforms, RepeatTransformations


class SVHNTransforms(Enum):
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.247, 0.243, 0.262)


class SVHNStandardTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.mean = SVHNTransforms.mean.value
        self.std = SVHNTransforms.std.value

    @property
    def train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def eval_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def query_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return transform


class SVHN(BaseData):
    def __init__(
            self,
            dataset_path: str,
            transforms: BaseTransforms = None,
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        self.transforms = SVHNStandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        torchvision.datasets.SVHN(self.dataset_path, split='train', download=True)
        torchvision.datasets.SVHN(self.dataset_path, split='test', download=True)
        # torchvision.datasets.SVHN(self.dataset_path, split='extra', download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='train', transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.SVHN(self.dataset_path, split='test', transform=self.eval_transform)


class SVHNPlain(SVHN):
    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None) -> None:
        super().__init__(dataset_path, PlainTransforms(), val_split, seed)


class SVHNContrastiveTransforms(SVHNStandardTransforms):
    def __init__(self, color_distortion_strength: float, rotation_probability: float):
        super().__init__()
        self._s = color_distortion_strength
        self.rotation_prob = rotation_probability

    @property
    def train_transform(self):
        color_jitter = torchvision.transforms.ColorJitter(0.8 * self._s, 0.8 * self._s, 0.8 * self._s, 0.2 * self._s)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(30)], self.rotation_prob),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            # GaussianBlur not used in original paper, however, other papers assign it importance
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=3)], p=0.5),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize(self.mean, self.std),
        ])
        return RepeatTransformations(transform)

    @property
    def eval_transform(self):
        return self.train_transform


class SVHNContrastive(SVHN):
    """
    Contrastive version of SVHN.

    This means that the transforms are repeated twice for each image, resulting in two views for each input image.
    """

    def __init__(self, dataset_path: str, val_split: float = 0.1, seed: int = None, cds=0.5, r_prob=0.33) -> None:
        super().__init__(dataset_path,
                         SVHNContrastiveTransforms(color_distortion_strength=cds, rotation_probability=r_prob),
                         val_split, seed)


def build_svhn(split, ds_path, mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970), return_info=False):
    warnings.warn('Deprecated method build_svhn.')
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    if split == 'train':
        ds = torchvision.datasets.SVHN(ds_path, split='train', download=True, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.SVHN(ds_path, split='train', download=True, transform=eval_transform)
    elif split == 'test':
        ds = torchvision.datasets.SVHN(ds_path, split='test', download=True, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
