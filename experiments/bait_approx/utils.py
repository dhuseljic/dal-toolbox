import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from rich.progress import track
from omegaconf import DictConfig

from dal_toolbox.models.laplace import LaplaceLayer, LaplaceModel
from dal_toolbox.datasets import CIFAR10, CIFAR100, Food101, STL10, ImageNet, TinyImageNet, Flowers102, StanfordDogs

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_dino_model(args):
    dino_model = torch.hub.load('facebookresearch/dinov2', args.dino_model_name)
    return dino_model


class DinoTransforms():
    def __init__(self, size=None, center_crop_size=224):
        if size:
            # https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L65-L70
            dino_mean = (0.485, 0.456, 0.406)
            dino_std = (0.229, 0.224, 0.225)
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) != 3 else x),
                transforms.Normalize(dino_mean, dino_std),
            ])

        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


def build_datasets(args):
    image_datasets = ['cifar10', 'stl10', 'cifar100', 'food101', 'flowers102',
                      'caltech101', 'stanford_dogs', 'tiny_imagenet', 'imagenet']
    text_datasets = ['agnews', 'dbpedia', 'clinc', 'trec']
    tabular_datasets = ['letter',  'aloi']

    if args.dataset_name in image_datasets:
        model = build_dino_model(args)
        data = build_image_data(args)
        data.train_dataset[0]
        train_ds = FeatureDataset(model, data.train_dataset, cache=True, cache_dir=args.feature_cache_dir)
        test_ds = FeatureDataset(model, data.test_dataset, cache=True, cache_dir=args.feature_cache_dir)
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
                                  cache_dir=args.feature_cache_dir, task="text")
        test_ds = FeatureDataset(model, data["test"], cache=True,
                                 cache_dir=args.feature_cache_dir, task="text")

    elif args.dataset_name in tabular_datasets:
        openml_id = {'letter': 6, 'aloi': 42396}
        train_ds, test_ds, num_classes = build_tabular_data(openml_id[args.dataset_name])
    else:
        raise NotImplementedError()

    return train_ds, test_ds, num_classes


def build_image_data(args):
    # transforms = PlainTransforms(resize=(224, 224))
    transforms = DinoTransforms(size=(256, 256))
    if args.dataset_name == 'cifar10':
        data = CIFAR10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'stl10':
        data = STL10(args.dataset_path, transforms=transforms)
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
        data = load_dataset("ag_news")  # maybe set cache_dir @denis cache_dir=
        num_classes = 4
    elif args.dataset_name == "dbpedia":
        data = load_dataset("dbpedia_14")  # maybe set cache_dir @denis cache_dir=
        data = data.rename_column("content", "text")
        num_classes = 14
    elif args.dataset_name == "clinc":
        data = load_dataset("clinc_oos", "plus")  # maybe set cache_dir @denis cache_dir=
        data = data.rename_column("intent", "label")
        num_classes = 151
    elif args.dataset_name == "trec":
        data = load_dataset("trec")  # maybe set cache_dir @denis cache_dir=
        num_classes = 6
        data = data.rename_column("coarse_label", "label")
    else:
        raise NotImplementedError()
    return data, num_classes


def build_tabular_data(data_id, path='data/'):
    X, y = fetch_openml(data_id=data_id, data_home=path, return_X_y=True)
    X = X.values
    y = y.values
    y = LabelEncoder().fit_transform(y)
    train, test = train_test_split(range(len(X)), random_state=0, test_size=0.25)
    scaler = StandardScaler().fit(X[train])

    X_train = torch.from_numpy(scaler.transform(X[train])).float()
    y_train = torch.from_numpy(y[train]).long()
    X_test = torch.from_numpy(scaler.transform(X[test])).float()
    y_test = torch.from_numpy(y[test]).long()
    num_classes = len(y_train.unique())
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test), num_classes


class LaplaceNet(LaplaceLayer):
    use_mean_field = True

    @torch.no_grad()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            if LaplaceNet.use_mean_field:
                # logits = self.forward_mean_field(inputs.to(device))
                logits = self(inputs.to(device))
            else:
                logits = self.forward_monte_carlo(inputs.to(device))
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
            inputs = batch[0].to(device)
            logits = self(inputs)

            features = inputs
            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)
            num_classes = logits.size(-1)

            factor = F.one_hot(max_indices, num_classes=num_classes) - probas
            embedding_batch = (factor[:, :, None] * features[:, None, :]).flatten(-2)

            embedding.append(embedding_batch.cpu())
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding

    @torch.no_grad()
    def get_topk_grad_representations(self, dataloader, device, topk, grad_likelihood='cross_entropy', normalize_top_probas=True):
        self.eval()
        self.to(device)

        embedding = []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = self(inputs)

            features = inputs
            probas = logits.softmax(-1)
            num_classes = logits.size(-1)
            probas_topk, top_preds = probas.topk(k=topk)

            if grad_likelihood == 'cross_entropy':
                factor = (torch.eye(num_classes, device=device)[:, None] - probas)
                batch_indices = torch.arange(len(top_preds)).unsqueeze(-1).expand(-1, top_preds.size(1))
                factor = factor[top_preds, batch_indices]

                embedding_batch = torch.einsum("njh,nd->njhd", factor, features).flatten(2)
                if normalize_top_probas:
                    probas_topk /= probas_topk.sum(-1, keepdim=True)
                embedding_batch = torch.sqrt(probas_topk)[:, :, None] * embedding_batch

            elif grad_likelihood == 'binary_cross_entropy':
                # We assume multiple independet binary cross entropy for highest probabilities
                if topk > 2:
                    raise ValueError('When using the binary cross entropy, topk must be 1 or 2.')
                max_probas = probas_topk[:, 0]
                factor = torch.eye(topk, device=device)[0] - max_probas[:, None]
                embedding_batch = torch.einsum("nk,nd->nkd", factor, features).flatten(2)

                probas_topk = torch.stack((max_probas, 1 - max_probas), dim=1)[:, :topk]
                if normalize_top_probas:
                    probas_topk /= probas_topk.sum(-1, keepdim=True)
                embedding_batch = torch.sqrt(probas_topk)[:, :, None] * embedding_batch

            elif grad_likelihood == 'cross_entropy_unbiased':

                cat = torch.distributions.Categorical(probas)
                factor = (torch.eye(num_classes, device=device)[:, None] - probas)
                batch_indices = torch.arange(len(top_preds)).unsqueeze(-1).expand(-1, top_preds.size(1))
                sampled_labels = cat.sample((topk,)).T
                factor = factor[sampled_labels, batch_indices]

                embedding_batch = torch.einsum("njh,nd->njhd", factor, features).flatten(2)
            else:

                raise NotImplementedError()

            embedding.append(embedding_batch.cpu())
        embedding = torch.cat(embedding)

        return embedding

    @torch.no_grad()
    def get_exp_grad_representations(self, dataloader, device, grad_likelihood='cross_entropy'):
        self.eval()
        self.to(device)

        embedding = []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = self(inputs)

            features = inputs
            probas = logits.softmax(-1)
            num_classes = logits.size(-1)

            if grad_likelihood == 'cross_entropy':
                factor = (torch.eye(num_classes, device=device)[:, None] - probas)
                embedding_batch = torch.einsum("jnh,nd->njhd", factor, features).flatten(2)
                embedding_batch = torch.sqrt(probas)[:, :, None] * embedding_batch

            elif grad_likelihood == 'binary_cross_entropy':
                max_probas = probas.max(dim=-1).values

                factor = torch.eye(2, device=device)[0] - max_probas[:, None]
                embedding_batch = torch.einsum("nk,nd->nkd", factor, features).flatten(2)
                probas_ = torch.stack((max_probas, 1 - max_probas), dim=1)
                embedding_batch = torch.sqrt(probas_)[:, :, None] * embedding_batch
            elif grad_likelihood == 'test':
                c = 10
                top_probas = probas.topk(c).values
                top_probas = top_probas / top_probas.sum(-1, keepdim=True)
                factor = (torch.eye(c, device=device)[:, None] - top_probas)
                embedding_batch = torch.einsum("jnh,nd->njhd", factor, features).flatten(2)

            else:
                raise NotImplementedError()

            embedding.append(embedding_batch.cpu())
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


class LaplaceMLP(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=128):
        super().__init__()
        self.l1 = nn.Linear(num_features, num_hidden)
        self.l2 = LaplaceNet(num_hidden, num_classes)
        self.act = nn.ReLU()

    def forward_feature(self, x):
        out = self.l1(x)
        out = self.act(out)
        return out

    def forward(self, x):
        feature = self.forward_feature(x)
        out = self.l2(feature)
        return out

    def forward_mean_field(self, x):
        feature = self.forward_feature(x)
        out = self.l2.forward_mean_field(feature)
        return out

    def forward_monte_carlo(self, x):
        feature = self.forward_feature(x)
        out = self.l2.forward_monte_carlo(feature)
        return out

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
            feature = self.forward_feature(inputs.to(device))
            all_representations.append(feature)
        representations = torch.cat(all_representations)
        return representations

    @torch.no_grad()
    def get_grad_representations(self, dataloader, device, grad_approx=True):
        self.eval()
        self.to(device)

        features = self.get_representations(dataloader, device)
        featureloader = DataLoader(TensorDataset(features), batch_size=dataloader.batch_size)
        embeddings = self.l2.get_grad_representations(featureloader, device=device)
        return embeddings

    @torch.no_grad()
    def get_topk_grad_representations(self, dataloader, device, topk, grad_likelihood='cross_entropy', normalize_top_probas=True):
        self.eval()
        self.to(device)

        features = self.get_representations(dataloader, device)
        featureloader = DataLoader(TensorDataset(features), batch_size=dataloader.batch_size)
        embeddings = self.l2.get_topk_grad_representations(
            featureloader, topk=topk, grad_likelihood=grad_likelihood, device=device,  normalize_top_probas=normalize_top_probas)
        return embeddings

    @torch.no_grad()
    def get_exp_grad_representations(self, dataloader, device, grad_likelihood='cross_entropy'):
        self.eval()
        self.to(device)

        features = self.get_representations(dataloader, device)
        featureloader = DataLoader(TensorDataset(features), batch_size=dataloader.batch_size)
        embeddings = self.l2.get_exp_grad_representations(
            featureloader, device=device, grad_likelihood=grad_likelihood)
        return embeddings


def build_model(args, **kwargs):
    num_features = kwargs['num_features']
    num_classes = kwargs['num_classes']

    # Laplace net because we want to be able to sample via Bayesian methods.
    # Not using the covariance is equivalent to deterministic model.
    if args.model.name == 'laplace':
        model = LaplaceNet(
            num_features,
            num_classes,
            mean_field_factor=args.model.mean_field_factor,
            mc_samples=args.model.mc_samples,
            bias=True,
        )
        if 'al' in args and args.al.strategy in ['bald', 'pseudo_bald', 'batch_bald']:
            LaplaceNet.use_mean_field = False
    elif args.model.name == 'laplace_mlp':
        model = LaplaceMLP(num_features, num_classes)
        if 'al' in args and args.al.strategy in ['bald', 'pseudo_bald', 'batch_bald']:
            LaplaceNet.use_mean_field = False
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

    model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
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
        for batch in track(dataloader, 'Inference'):  # change
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
