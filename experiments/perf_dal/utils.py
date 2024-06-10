import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from omegaconf import DictConfig

from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import DinoTransforms, FeatureDataset
from dal_toolbox.datasets import CIFAR10, CIFAR100, Food101, STL10, Snacks, DTD, Flowers102, TinyImageNet
from dal_toolbox.datasets import ImageNet, StanfordDogs


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
        data = CIFAR10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stl10':
        data = STL10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'snacks':
        data = Snacks(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'dtd':
        data = DTD(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'cifar100':
        data = CIFAR100(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'food101':
        data = Food101(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'flowers102':
        data = Flowers102(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stanford_dogs':
        data = StanfordDogs(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'tiny_imagenet':
        data = TinyImageNet(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'imagenet':
        data = ImageNet(args.dataset_path, transforms=transforms)
    else:
        raise NotImplementedError()
    return data


def build_text_data(args):
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

class FeatureDataset:

    def __init__(self, model, dataset, cache=False, cache_dir=None, batch_size=256, device='cuda', task=None):
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
                features, labels = self.get_features(model, dataset, batch_size, device, task)  # change
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
                sample = dataset["input_ids"][0]
            hasher.update(str(sample).encode())
        return hasher.hexdigest()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def get_features(self, model, dataset, batch_size, device, task=None):
        print('Getting ssl features..')
        dataloader = DataLoader(dataset, batch_size=batch_size)
        features = []
        labels = []
        model.eval()
        model.to(device)
        for batch in dataloader:  # change
            if task == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features.append(model(input_ids, attention_mask).to('cpu'))
                labels.append(batch["label"])
            else:
                features.append(model(batch[0].to(device)).to('cpu'))
                labels.append(batch[-1])

        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels


class BertSequenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertSequenceClassifier, self).__init__()

        self.num_classes = num_classes
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask, labels=None, output_hidden_states=True)
        logits = outputs['logits']

        # huggingface takes pooler output for classification (not accessible here anymore, would need bert model)
        last_hidden_state = outputs['hidden_states'][-1]  # (batch, sequence, dim)
        # (batch, dim)     #not in bert, taken from distilbert and roberta
        cls_state = last_hidden_state[:, 0, :]
        return cls_state

def flatten_cfg(cfg, parent_key='', sep='.'):
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)