import torchvision
from torchvision import transforms


def build_svhn(split, ds_path, mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970), return_info=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
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
