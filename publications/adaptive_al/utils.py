import torch
import torch.nn as nn

from dal_toolbox.models.laplace import LaplaceLinear, LaplaceModel
from dal_toolbox import datasets as dal_datasets
from dal_toolbox.datasets.utils import DinoTransforms, FeatureDataset


image_datasets = ['cifar10', 'stl10', 'snacks', 'dtd', 'cifar100', 'food101', 'flowers102',
                  'caltech101', 'stanford_dogs', 'tiny_imagenet', 'imagenet']
text_datasets = ['agnews', 'dbpedia', 'banking77', 'clinc']


def build_backbone(args):
    if args.dataset.backbone == 'dinov2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else:
        raise NotImplementedError(f'Backbone {args.dataset.backbone} not implemented.')
    return backbone


def build_datasets(args):
    backbone = build_backbone(args)

    if args.dataset.name in image_datasets:
        data = build_image_data(args)
        if args.dataset.cache_features:
            train_ds = FeatureDataset(backbone, data.train_dataset, cache=True, cache_dir=args.dataset.path)
            test_ds = FeatureDataset(backbone, data.test_dataset, cache=True, cache_dir=args.dataset.path)
        else:
            train_ds = data.train_dataset
            test_ds = data.test_dataset
        num_classes = data.num_classes
    return train_ds, test_ds, num_classes


def build_image_data(args):
    transforms = DinoTransforms(size=(256, 256))
    if args.dataset.name == 'cifar10':
        data = dal_datasets.CIFAR10(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'stl10':
        data = dal_datasets.STL10(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'snacks':
        data = dal_datasets.Snacks(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'dtd':
        data = dal_datasets.DTD(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'cifar100':
        data = dal_datasets.CIFAR100(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'food101':
        data = dal_datasets.Food101(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'flowers102':
        data = dal_datasets.Flowers102(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'stanford_dogs':
        data = dal_datasets.StanfordDogs(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'tiny_imagenet':
        data = dal_datasets.TinyImageNet(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'imagenet':
        data = dal_datasets.ImageNet(args.dataset.path, transforms=transforms)
    else:
        raise NotImplementedError()
    return data


def build_text_data(args):
    from datasets import load_dataset  # Huggingface import
    if args.dataset_name == "agnews":
        data = load_dataset("ag_news")
        num_classes = 4
    elif args.dataset_name == "dbpedia":
        data = load_dataset("dbpedia_14")
        data = data.rename_column("content", "text")
        num_classes = 14
    elif args.dataset_name == "banking77":
        data = load_dataset("banking77")
        num_classes = 77
        # data = data.rename_column("coarse_label", "label")
    elif args.dataset_name == "clinc":
        data = load_dataset("clinc_oos", "plus")
        data = data.rename_column("intent", "label")
        num_classes = 151
    else:
        raise NotImplementedError()
    return data, num_classes


def build_model(args, num_features, num_classes):
    laplace_kwargs = dict(mean_field_factor=args.model.mean_field_factor,
                          mc_samples=args.model.mc_samples, bias=True)
    if args.model.name == 'linear':
        model = LinearModel(num_features, num_classes, **laplace_kwargs)
    elif args.model.name == 'mlp':
        model = MLP(num_features, num_classes, **laplace_kwargs)
    elif args.model.name == 'all':
        raise NotImplementedError(f"Training of {args.model.name} not implemented.")
    else:
        raise NotImplementedError(f"Training of {args.model.name} not implemented.")

    params = [{'params': [p for n, p in model.named_parameters()]}]

    if args.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.optimizer.lr, nesterov=args.optimizer.nesterov,
                                    momentum=args.optimizer.momentum, weight_decay=args.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.model.name} not implemented.")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.model.num_epochs)

    model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return model


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, **laplace_kwargs):
        super().__init__()
        self.layer = LaplaceLinear(in_features, out_features, **laplace_kwargs)

    def forward_features(self, x):
        return x

    def forward_head(self, x, mean_field=False):
        if mean_field:
            out = self.layer.forward_mean_field(x)
        else:
            out = self.layer(x)
        return out

    def forward_mean_field(self, x):
        return self.layer.forward_mean_field(x)

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.forward_head(features)
        return logits


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden=512, **laplace_kwargs):
        super().__init__()
        self.layer1 = nn.Linear(in_features, num_hidden)
        self.layer2 = LaplaceLinear(num_hidden, out_features, **laplace_kwargs)
        self.act = nn.ReLU()

    def forward_features(self, x):
        out = self.layer1(x)
        out = self.act(out)
        return out

    def forward_head(self, x, mean_field=False):
        if mean_field:
            out = self.layer2.forward_mean_field(x)
        else:
            out = self.layer2(x)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.forward_head(features)
        return logits

    def forward_mean_field(self, x):
        features = self.forward_features(x)
        mean_field_logits = self.forward_head(features, mean_field=True)
        return mean_field_logits


def flatten_cfg(cfg, parent_key='', sep='.'):
    from omegaconf import DictConfig
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
