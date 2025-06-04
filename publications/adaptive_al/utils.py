import torch
from dal_toolbox import datasets as dal_datasets
from dal_toolbox.datasets.utils import DinoTransforms, FeatureDataset


def build_datasets(args, val_split=False, cache_features=True):
    image_datasets = ['cifar10', 'stl10', 'snacks', 'dtd', 'cifar100', 'food101', 'flowers102',
                      'caltech101', 'stanford_dogs', 'tiny_imagenet', 'imagenet']
    text_datasets = ['agnews', 'dbpedia', 'banking77', 'clinc']

    if args.dataset_name in image_datasets:
        data = build_image_data(args)
        if cache_features:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

            train_ds = FeatureDataset(model, data.train_dataset, cache=True, cache_dir=args.dataset_path)
            if val_split:
                test_ds = FeatureDataset(model, data.val_dataset, cache=True, cache_dir=args.dataset_path)
            else:
                test_ds = FeatureDataset(model, data.test_dataset, cache=True, cache_dir=args.dataset_path)
        else:
            train_ds = data.train_dataset
            if val_split:
                test_ds = data.val_dataset
            else:
                test_ds = data.test_dataset
        num_classes = data.num_classes

    elif args.dataset_name in text_datasets:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        data, num_classes = build_text_data(args)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        data = data.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            ),
            batched=True,
            batch_size=1000)

        data = data.remove_columns(
            list(set(data['train'].column_names)-set(['input_ids', 'attention_mask', 'label'])))
        data = data.with_format("torch")

        model = BertSequenceClassifier(num_classes=num_classes)
        train_ds = FeatureDataset(model, data["train"], cache=True,
                                  cache_dir=args.dataset_path, task="text")
        test_ds = FeatureDataset(model, data["test"], cache=True,
                                 cache_dir=args.dataset_path, task="text")

    return train_ds, test_ds, num_classes


def build_image_data(args):
    transforms = DinoTransforms(size=(256, 256))
    if args.dataset_name == 'cifar10':
        data = dal_datasets.CIFAR10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stl10':
        data = dal_datasets.STL10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'snacks':
        data = dal_datasets.Snacks(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'dtd':
        data = dal_datasets.DTD(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'cifar100':
        data = dal_datasets.CIFAR100(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'food101':
        data = dal_datasets.Food101(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'flowers102':
        data = dal_datasets.Flowers102(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stanford_dogs':
        data = dal_datasets.StanfordDogs(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'tiny_imagenet':
        data = dal_datasets.TinyImageNet(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'imagenet':
        data = dal_datasets.ImageNet(args.dataset_path, transforms=transforms)
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
    if args.model.name == 'linear':
        pass
    elif args.model.name == 'mlp':
        pass
    elif args.model.name == 'all':
        pass
    else:
        raise NotImplementedError(f"Training of {args.model.name} not implemented.")
    return model


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
