import os
import hydra
import torch
import mlflow
from lightning import Trainer
from dal_toolbox.datasets import CIFAR10, CIFAR100, SVHN
from dal_toolbox.datasets.utils import PlainTransforms
from dal_toolbox.models.sngp import RandomFeatureGaussianProcess, SNGPModel
from dal_toolbox.metrics import Accuracy, AdaptiveCalibrationError, OODAUROC, OODAUPR, entropy_from_logits
from dal_toolbox.utils import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path="./configs", config_name="ablation")
def main(args):
    seed_everything(1)  # seed for val split being identical each time

    # First fixed seed for datasets to be identical
    mlflow.set_tracking_uri(uri="file://{}".format(os.path.abspath(args.mlflow_dir)))
    mlflow.set_experiment("Ablation")
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    print(OmegaConf.to_yaml(args))

    # Setup
    dino_model = build_dino_model(args)
    data = build_data(args)
    ood_data = build_ood_data(args)

    train_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.train_dataset,
                                  normalize_features=True, cache=True)
    test_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.test_dataset, normalize_features=True, cache=True)
    ood_ds = DinoFeatureDataset(dino_model=dino_model, dataset=ood_data.test_dataset,
                                normalize_features=True, cache=True)

    seed_everything(args.random_seed)

    model = build_model(args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)

    # Train
    train_indices = torch.randperm(len(train_ds))[:args.num_train_samples]
    train_loader = DataLoader(
        Subset(train_ds, indices=train_indices),
        batch_size=args.model.train_batch_size,
        shuffle=True,
        drop_last=len(train_indices) > args.model.train_batch_size,
    )
    trainer = Trainer(
        max_epochs=args.model.num_epochs
    )
    trainer.fit(model, train_dataloaders=train_loader)

    # Eval
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size, shuffle=False)
    predictions = trainer.predict(model, test_loader)
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])
    test_entropies = entropy_from_logits(test_logits)

    ood_loader = DataLoader(ood_ds, batch_size=args.model.predict_batch_size, shuffle=False)
    predictions = trainer.predict(model, ood_loader)
    ood_logits = torch.cat([pred[0] for pred in predictions])
    ood_entropies = entropy_from_logits(ood_logits)

    test_stats = {
        'accuracy': Accuracy()(test_logits, test_labels).item(),
        'ACE': AdaptiveCalibrationError()(test_logits, test_labels).item(),
        'AUROC': OODAUROC()(test_entropies, ood_entropies).item(),
        'AUPR': OODAUPR()(test_entropies, ood_entropies).item(),
    }

    print(test_stats)
    mlflow.log_metrics(test_stats)
    mlflow.end_run()


def build_dino_model(args):
    dino_model = torch.hub.load(
        'facebookresearch/dinov2', args.dino_model_name)
    return dino_model


def build_data(args):
    transforms = PlainTransforms(resize=(224, 224))
    if args.dataset_name == 'cifar10':
        data = CIFAR10(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'cifar100':
        data = CIFAR100(args.dataset_path, transforms=transforms)
    elif args.dataset_name == 'svhn':
        data = SVHN(args.dataset_path, transforms=transforms)
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
    else:
        raise NotImplementedError()
    return data


def build_model(args, **kwargs):
    num_features = kwargs['num_features']
    num_classes = kwargs['num_classes']
    model = RandomFeatureGaussianProcess(
        in_features=num_features,
        out_features=num_classes,
        num_inducing=args.model.num_inducing,
        kernel_scale=args.model.kernel_scale,
        scale_random_features=args.model.scale_random_features,
        optimize_kernel_scale=args.model.optimize_kernel_scale,
        mean_field_factor=args.model.mean_field_factor,
    )
    if args.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optimizer.lr,
                                    momentum=args.optimizer.momentum, weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer.lr,
                                      weight_decay=args.optimizer.weight_decay)
    else:
        raise NotImplementedError()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.model.num_epochs)

    train_metrics = {}
    val_metrics = {}
    model = SNGPModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics
    )
    return model


class DinoFeatureDataset:

    def __init__(self, dino_model, dataset, normalize_features=True, cache=False, device='cuda'):

        if cache:
            home_dir = os.path.expanduser('~')
            dino_cache_dir = os.path.join(home_dir, '.cache', 'dino_features')
            os.makedirs(dino_cache_dir, exist_ok=True)
            hash = self.create_hash_from_dataset_and_model(dataset, dino_model)
            file_name = os.path.join(dino_cache_dir, hash + '.pth')
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


def flatten_cfg(cfg, parent_key='', sep='.'):
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == '__main__':
    main()
