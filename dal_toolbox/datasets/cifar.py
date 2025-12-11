import numpy as np
import torchvision
from .corruptions import RandAugment

from PIL import Image
from .base import BaseData, BaseTransforms
from .corruptions import GaussianNoise
from .transforms import StandardTransforms, MultiTransform

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.262)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class CIFAR10StandardTransforms(StandardTransforms):
    def __init__(self):
        super().__init__(resize=32, mean=CIFAR10_MEAN, std=CIFAR10_STD)


class CIFAR10(BaseData):
    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        self.transforms = CIFAR10StandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        torchvision.datasets.CIFAR10(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR10(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.CIFAR10(self.dataset_path, train=False, transform=self.eval_transform)


class CIFAR10CTransforms(CIFAR10StandardTransforms):
    def __init__(self, severity: float):
        super().__init__()
        self.severity = severity

    @property
    def eval_transform(self):
        eval_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            GaussianNoise(self.severity)
        ])
        return eval_transform


class CIFAR10C(CIFAR10):
    def __init__(self,
                 dataset_path: str,
                 severity: float = .5,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        super().__init__(dataset_path, CIFAR10CTransforms(severity=severity), val_split, seed)


class CIFAR100StandardTransforms(StandardTransforms):
    def __init__(self):
        super().__init__(resize=32, mean=CIFAR100_MEAN, std=CIFAR100_STD)


class CIFAR100(BaseData):
    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 seed: int = None) -> None:
        self.transforms = CIFAR100StandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 100

    def download_datasets(self):
        torchvision.datasets.CIFAR100(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR100(self.dataset_path, train=False, download=True)

    @property
    def full_train_dataset(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return torchvision.datasets.CIFAR100(self.dataset_path, train=False, transform=self.eval_transform)


class CIFAR10LT(BaseData):
    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.1,
                 imbalance_ratio: float = 0.01,
                 seed: int = None) -> None:
        self.transforms = CIFAR10StandardTransforms() if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform
        self.imbalance_ratio = imbalance_ratio
        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10

    def download_datasets(self):
        _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=True, download=True)
        _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=False, download=True)

    @property
    def full_train_dataset(self):
        return _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=True, transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=True, transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=True, transform=self.query_transform)

    @property
    def test_dataset(self):
        return _CIFAR10LT(self.dataset_path, self.imbalance_ratio, train=False, transform=self.eval_transform)


class _CIFAR10LT(torchvision.datasets.CIFAR10):
    # Ref: https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/data/ImbalanceCIFAR.py
    cls_num = 10

    def __init__(self, root, imbalance_ratio, imb_type='exp', train=True, transform=None, download=False):
        super(_CIFAR10LT, self).__init__(root, train, transform=None, target_transform=None, download=download)
        self.train = train
        self.transform = transform
        if self.train:
            img_num_list = self.get_img_num_per_cls(10, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)
        self.labels = self.targets

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR10PseudoLabelTransforms(CIFAR10StandardTransforms):
    def __init__(self):
        super().__init__()

    @property
    def train_transform(self):
        transform_weak = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform_weak


class CIFAR10PIModelTransforms(CIFAR10StandardTransforms):
    @property
    def train_transform(self):
        transform_weak1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        transform_weak2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return MultiTransform(transform_weak1, transform_weak2)


class CIFAR10FixMatchTransforms(CIFAR10StandardTransforms):
    @property
    def train_transform(self):
        transform_weak = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])

        transform_strong = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return MultiTransform(transform_weak, transform_strong)
