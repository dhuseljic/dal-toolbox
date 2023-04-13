import torch
import torchvision
from torchvision import transforms
from .corruptions import RandAugment

class BasicSSLDataset(torchvision.datasets.VisionDataset):
    def __init__(self, ds_path, crop_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262)):
        super().__init__(ds_path)
        self.ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True)
        self.mean = mean
        self.std = std
        self.transforms_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
        ])
        self.transforms_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std)
    ])
    
    # TODO: Is this better? train one epochs than need to be adapted but could be unified even more
    #def __getitem__(self, idx):
    #    X, y = self.ds.__getitem__(idx)
    #    return self.transforms_weak(X), self.transforms_weak(X), self.transforms_strong(X), y, idx


class PseudoLabelWrapper(BasicSSLDataset):
    def __getitem__(self, idx):
        X, y = self.ds.__getitem__(idx)
        return self.transforms_weak(X), y
    
class PiModelWrapper(BasicSSLDataset):
    def __getitem__(self, idx):
        X, y = self.ds.__getitem__(idx)
        return self.transforms_weak(X), self.transforms_weak(X), y

class FixMatchWrapper(BasicSSLDataset):
    def __getitem__(self, idx):
        X, y = self.ds.__getitem__(idx)
        return self.transforms_weak(X), self.transforms_strong(X), y

class FlexMatchWrapper(BasicSSLDataset):
    def __getitem__(self, idx):
        X, y = self.ds.__getitem__(idx)
        return self.transforms_weak(X), self.transforms_strong(X), y, idx
    
    

def build_ssl_dataset(ssl_algorithm, ds_path):
    if ssl_algorithm == 'pseudo_labels':
        ds = PseudoLabelWrapper(ds_path=ds_path)
    elif ssl_algorithm == 'pi_model':
        ds = PiModelWrapper(ds_path=ds_path)
    elif ssl_algorithm == 'fixmatch':
        ds = FixMatchWrapper(ds_path=ds_path)
    elif ssl_algorithm == 'flexmatch':
        ds = FlexMatchWrapper(ds_path=ds_path)
    else:
        assert True, 'algorithm not kown'
    return ds