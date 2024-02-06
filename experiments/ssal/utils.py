import os
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from omegaconf import DictConfig

from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.sngp import RandomFeatureGaussianProcess, SNGPModel
from dal_toolbox.models.laplace import LaplaceLayer, LaplaceModel

from dal_toolbox.datasets import CIFAR10, CIFAR100, SVHN, Food101, STL10
from dal_toolbox.datasets.utils import PlainTransforms


def build_dino_model(args):
    dino_model = torch.hub.load('facebookresearch/dinov2', args.dino_model_name)

    #  simclr_checkpoint = torch.load(path / 'wide_resnet_28_10_CIFAR10_0.907.pth', map_location='cpu')
    #  encoder = wide_resnet.wide_resnet_28_10(num_classes=1, dropout_rate=0.3)
    #  encoder.linear = nn.Identity()
    #  encoder.load_state_dict(simclr_checkpoint['model'])
    #  encoder.requires_grad_(False)
    return dino_model


def build_data(args):
    transforms = PlainTransforms(resize=(224, 224))
    if args.dataset_name == 'cifar10':
        data = CIFAR10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'cifar100':
        data = CIFAR100(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'svhn':
        data = SVHN(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'food101':
        data = Food101(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stl10':
        data = STL10(args.dataset_path, transforms=transforms)
    else:
        raise NotImplementedError()
    return data


def build_ood_data(args):
    transforms = PlainTransforms(resize=(224, 224))
    if args.ood_dataset_name == 'cifar10':
        data = CIFAR10(args.dataset_path, transforms=transforms)
    elif args.ood_dataset_name == 'cifar100':
        data = CIFAR100(args.dataset_path, transforms=transforms)
    elif args.ood_dataset_name == 'svhn':
        data = SVHN(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'food101':
        data = Food101(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stl10':
        data = STL10(args.dataset_path, transforms=transforms)
    else:
        raise NotImplementedError()
    return data


class SNGPNet(RandomFeatureGaussianProcess):
    @torch.no_grad()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self.forward_mean_field(inputs.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

    @torch.no_grad()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_representations = []
        for batch in dataloader:
            inputs = batch[0]
            # TODO: which representation to use? RFF or input
            all_representations.append(inputs)
        representations = torch.cat(all_representations)
        return representations


class LaplaceNet(LaplaceLayer):
    @torch.no_grad()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self.forward_mean_field(inputs.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

    @torch.no_grad()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_representations = []
        for batch in dataloader:
            inputs = batch[0]
            all_representations.append(inputs)
        representations = torch.cat(all_representations)
        return representations

    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device):
        self.eval()
        self.to(device)

        embedding = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self.forward(inputs.to(device))

            features = inputs.cpu()
            logits = logits.cpu()
            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            num_samples, feature_dim = inputs.shape
            num_classes = logits.size(-1)

            # for each sample in a batch and for each class, compute the gradient wrt to weights
            embedding_batch = torch.empty([num_samples, feature_dim * num_classes])
            for n in range(len(inputs)):
                for c in range(num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding


class DeterministcNet(nn.Linear):
    @torch.no_grad()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self(inputs.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

    @torch.no_grad()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_representations = []
        for batch in dataloader:
            inputs = batch[0]
            all_representations.append(inputs)
        representations = torch.cat(all_representations)
        return representations


def build_model(args, **kwargs):
    num_features = kwargs['num_features']
    num_classes = kwargs['num_classes']
    if args.model.name == 'sngp':
        model = SNGPNet(
            in_features=num_features,
            out_features=num_classes,
            num_inducing=args.model.num_inducing,
            kernel_scale=args.model.kernel_scale,
            scale_random_features=args.model.scale_random_features,
            optimize_kernel_scale=args.model.optimize_kernel_scale,
            mean_field_factor=args.model.mean_field_factor,
        )
    elif args.model.name == 'laplace':
        model = LaplaceNet(num_features, num_classes, mean_field_factor=args.model.mean_field_factor)
    elif args.model.name == 'deterministic':
        model = DeterministcNet(num_features, num_classes)
    else:
        raise NotImplementedError()

    if args.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optimizer.lr,
                                    momentum=args.optimizer.momentum, weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.lr,
                                     weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer.lr,
                                      weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.optimizer.lr,
                                      weight_decay=args.optimizer.weight_decay)
    else:
        raise NotImplementedError()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.model.num_epochs)

    if args.model.name == 'sngp':
        model = SNGPModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'laplace':
        model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'deterministic':
        model = DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    else:
        raise NotImplementedError()
    return model


def flatten_cfg(cfg, parent_key='', sep='.'):
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DinoFeatureDataset:

    def __init__(self, dino_model, dataset, normalize_features=True, cache=False, cache_dir=None, device='cuda'):

        if cache:
            if cache_dir is None:
                home_dir = os.path.expanduser('~')
                cache_dir = os.path.join(home_dir, '.cache', 'dino_features')
            os.makedirs(cache_dir, exist_ok=True)
            hash = self.create_hash_from_dataset_and_model(dataset, dino_model)
            file_name = os.path.join(cache_dir, hash + '.pth')
            if os.path.exists(file_name):
                print('Loading cached features from', file_name)
                features, labels = torch.load(file_name, map_location='cpu')
            else:
                features, labels = self.get_dino_features(dino_model, dataset, device)
                print('Saving features to cache file', file_name)
                torch.save((features, labels), file_name)
        else:
            features, labels = self.get_dino_features(dino_model, dataset, device)

        if normalize_features:
            features_mean = features.mean(0)
            features_std = features.std(0) + 1e-9
            features = (features - features_mean) / features_std

        self.features = features
        self.labels = labels

    def create_hash_from_dataset_and_model(self, dataset, dino_model, num_hash_samples=50):
        import hashlib
        hasher = hashlib.md5()

        num_samples = len(dataset)
        hasher.update(str(num_samples).encode())
        num_parameters = sum([p.numel() for p in dino_model.parameters()])
        hasher.update(str(num_parameters).encode())

        indices_to_hash = range(0, num_samples, num_samples//num_hash_samples)
        for idx in indices_to_hash:
            sample = dataset[idx][0]
            hasher.update(str(sample).encode())
        return hasher.hexdigest()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def get_dino_features(self, dino_model, dataset, device):
        print('Getting dino features..')
        dataloader = DataLoader(dataset, batch_size=256, num_workers=4)

        features = []
        labels = []
        dino_model.to(device)
        for batch in tqdm(dataloader):
            features.append(dino_model(batch[0].to(device)).to('cpu'))
            labels.append(batch[-1])
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels
