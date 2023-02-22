import torchvision
from .presets import ClassificationPresetTrain, ClassificationPresetEval


def build_imagenet(split, ds_path, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), return_info=False):
    train_transform = ClassificationPresetTrain(crop_size=224, mean=mean, std=std)
    eval_transform = ClassificationPresetEval(crop_size=224, mean=mean, std=std)
    if split == 'train':
        ds = torchvision.datasets.ImageNet(root=ds_path, split=split, transform=train_transform)
    elif split == 'query':
        ds = torchvision.datasets.ImageNet(root=ds_path, split=split, transform=eval_transform)
    elif split == 'val':
        ds = torchvision.datasets.ImageNet(root=ds_path, split=split, transform=eval_transform)
    if return_info:
        ds_info = {'n_classes': 1000, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds