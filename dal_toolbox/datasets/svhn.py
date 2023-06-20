import warnings 

from torchvision import transforms, datasets
from .base import AbstractData


class SVHN(AbstractData):
    def __init__(
            self,
            dataset_path: str,
            mean: tuple = (0.4377, 0.4438, 0.4728),
            std: tuple = (0.1980, 0.2010, 0.1970),
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        self.mean = mean
        self.std = std
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        datasets.SVHN(self.dataset_path, split='train', download=True)
        datasets.SVHN(self.dataset_path, split='test', download=True)
        # datasets.SVHN(self.dataset_path, split='extra', download=True)

    @property
    def full_train_dataset(self):
        return datasets.SVHN(self.dataset_path, split='train', transform=self.train_transforms)

    @property
    def full_train_dataset_eval_transforms(self):
        return datasets.SVHN(self.dataset_path, split='train', transform=self.eval_transforms)

    @property
    def full_train_dataset_query_transforms(self):
        return datasets.SVHN(self.dataset_path, split='train', transform=self.query_transforms)

    @property
    def test_dataset(self):
        return datasets.SVHN(self.dataset_path, split='test', transform=self.eval_transforms)

    @property
    def train_transforms(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transform

    @property
    def eval_transforms(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform

    @property
    def query_transforms(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform


def build_svhn(split, ds_path, mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970), return_info=False):
    warnings.warn('Deprecated method build_svhn.')
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == 'train':
        ds = datasets.SVHN(ds_path, split='train', download=True, transform=train_transform)
    elif split == 'query':
        ds = datasets.SVHN(ds_path, split='train', download=True, transform=eval_transform)
    elif split == 'test':
        ds = datasets.SVHN(ds_path, split='test', download=True, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
