import os
import torch
import logging

from torch.utils.data import DataLoader
from torchvision import transforms

def build_dino_model(args):
    dino_model = torch.hub.load('facebookresearch/dinov2', args.model.dino_model_name)
    return dino_model


class DinoTransforms():
    def __init__(self, size=None, center_crop_size=224, finetune=False):
        # https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L65-L70
        self.finetune = finetune
        dino_mean = (0.485, 0.456, 0.406)
        dino_std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(center_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(dino_mean, dino_std)
        ])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class FeatureDataset:
    def __init__(self, model, dataset, cache=False, cache_dir=None, batch_size=256, device='cuda'):
        if cache:
            if cache_dir is None:
                home_dir = os.path.expanduser('~')
                cache_dir = os.path.join(home_dir, '.cache', 'feature_datasets')
            os.makedirs(cache_dir, exist_ok=True)
            hash = self.create_hash_from_dataset_and_model(dataset, model)

            file_name = os.path.join(cache_dir, hash + '.pth')
            if os.path.exists(file_name):
                print('Loading cached features from', file_name)
                features, labels = torch.load(file_name, map_location='cpu')
            else:
                features, labels = self.get_features(model, dataset, batch_size, device)
                print('Saving features to cache file', file_name)
                torch.save((features, labels), file_name)
        else:
            features, labels = self.get_features(model, dataset, batch_size, device)

        self.features = features
        self.labels = labels

    def create_hash_from_dataset_and_model(self, dataset, dino_model, num_hash_samples=50):
        import hashlib
        hasher = hashlib.md5()

        num_samples = len(dataset)
        hasher.update(str(num_samples).encode())

        num_parameters = sum([p.numel() for p in dino_model.parameters()])
        hasher.update(str(dino_model).encode())
        hasher.update(str(num_parameters).encode())

        indices_to_hash = range(0, num_samples, num_samples//num_hash_samples)
        for idx in indices_to_hash:
            # change for text
            try:
                sample = dataset[idx][0]
            except:
                sample = dataset.dataset.data[0]
            hasher.update(str(sample).encode())
        return hasher.hexdigest()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def get_features(self, model, dataset, batch_size, device):
        logging.info('Getting ssl features..')
        dataloader = DataLoader(dataset, batch_size=batch_size)
        features = []
        labels = []
        model.eval()
        model.to(device)
        log_interval = len(dataloader) // 25
        for i, batch in enumerate(dataloader):
            features.append(model(batch[0].to(device)).to('cpu'))
            labels.append(batch[-1])
            if i % log_interval == 0:
                logging.info(f"Getting ssl features...[{round(100*i/len(dataloader))}%]")
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels