import torchvision
from torchvision import transforms

def build_fashionmnist(split, ds_path, mean=(0.5,), std=(0.5,), return_info=False):
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    if split == 'train':
        ds = torchvision.datasets.FashionMNIST(ds_path, train=True, download=True, transform=transform)
    elif split == 'query':
        ds = torchvision.datasets.FashionMNIST(ds_path, train=True, download=True, transform=transform)
    else:
        ds = torchvision.datasets.FashionMNIST(ds_path, train=False, download=True, transform=transform)
    if return_info:
        ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
        return ds, ds_info
    return ds
