import torchvision

class BasicSSLWrapper(torchvision.datasets.VisionDataset):
    def __init__(self, ds, ds_path, transforms_weak=None, transforms_strong=None):
        super().__init__(ds_path)
        self.ds = ds
        self.transforms_weak = transforms_weak
        self.transforms_strong = transforms_strong
    
    def __len__(self) -> int:
        return len(self.ds)

class PseudoLabelWrapper(BasicSSLWrapper):
    def __getitem__(self, idx):
        X, y = self.ds[idx]
        return self.transforms_weak(X), y
    
class PiModelWrapper(BasicSSLWrapper):
    def __getitem__(self, idx):
        X, y = self.ds[idx]
        return self.transforms_weak(X), self.transforms_weak(X), y

class FixMatchWrapper(BasicSSLWrapper):
    def __getitem__(self, idx):
        X, y = self.ds[idx]
        return self.transforms_weak(X), self.transforms_strong(X), y

class FlexMatchWrapper(BasicSSLWrapper):
    def __getitem__(self, idx):
        X, y = self.ds[idx]
        return self.transforms_weak(X), self.transforms_strong(X), y, idx