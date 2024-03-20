from torch.utils.data import Dataset
from .base import BaseData, BaseTransforms
from .utils import PlainTransforms
from datasets import load_dataset, config


class TinyImageNet(BaseData):

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
        return 200

    def download_datasets(self):
        _TinyImageNet(self.dataset_path, split='train')
        _TinyImageNet(self.dataset_path, split='test')

    @property
    def full_train_dataset(self):
        return _TinyImageNet(self.dataset_path, split="train", transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return _TinyImageNet(self.dataset_path, split="train", transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return _TinyImageNet(self.dataset_path, split="train", transform=self.query_transform)

    @property
    def test_dataset(self):
        return _TinyImageNet(self.dataset_path, split="test", transform=self.eval_transform)


class _TinyImageNet(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        split_mapping = {'train': 'train', 'test': 'valid'}

        config.DOWNLOADED_DATASETS_PATH = root
        self.ds = load_dataset('Maysee/tiny-imagenet', split=split_mapping[self.split])

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
        

