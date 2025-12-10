import os
from torchvision import datasets
from .base import BaseData, BaseTransforms
from .transforms import CustomTransforms, TransformPresetTrain, TransformPresetEval

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetStandardTransforms(CustomTransforms):
    def __init__(self):
        train_transform = TransformPresetTrain(crop_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        eval_transform = TransformPresetEval(crop_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        super().__init__(train_transform=train_transform, eval_transform=eval_transform)


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
