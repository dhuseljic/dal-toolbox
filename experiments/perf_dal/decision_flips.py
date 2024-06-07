import os
import hydra
import mlflow
import logging
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from lightning import Trainer
from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import DinoTransforms, FeatureDataset
from dal_toolbox.models.laplace import LaplaceModel, LaplaceLayer
from dal_toolbox.utils import seed_everything
logging.getLogger("lightning").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="./configs", config_name="decision_flips")
def main(args):
    seed_everything(args.random_seed)
    transforms = DinoTransforms(size=(256, 256))
    data = CIFAR10('/home/datasets', transforms=transforms)
    train_ds = data.train_dataset
    test_ds = data.test_dataset

    ssl_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    train_ds = FeatureDataset(ssl_model, train_ds, cache=True)
    test_ds = FeatureDataset(ssl_model, test_ds, cache=True)

    trainer_kwargs = dict(
        max_epochs=args.num_epochs,
        enable_checkpointing=False,
        logger=False,
        barebones=True,
    )
    model = LaplaceLayer(384, 10)
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=trainer_kwargs['max_epochs'])
    lit_model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)

    train_indices = np.random.permutation(len(train_ds))
    test_indices = np.random.choice(range(len(test_ds)), size=args.num_test_samples, replace=False)
    test_loader = DataLoader(test_ds, batch_size=32, sampler=test_indices)

    results = []
    for num_new in range(1, args.max_new_samples+1):

        # Training
        lit_model.reset_states()
        trainer = Trainer(**trainer_kwargs)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=train_indices[:args.num_train_samples])
        trainer.fit(lit_model, train_loader)
        test_predictions = trainer.predict(lit_model, test_loader)

        lit_model.reset_states()
        trainer = Trainer(**trainer_kwargs)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=train_indices[:args.num_train_samples+num_new])
        trainer.fit(lit_model, train_loader)
        test_predictions_new = trainer.predict(lit_model, test_loader)

        # Eval
        test_logits = torch.cat([pred[0] for pred in test_predictions])
        test_labels = torch.cat([pred[1] for pred in test_predictions])
        test_decisions = test_logits.argmax(-1)
        test_acc = torch.mean((test_decisions == test_labels).float())

        test_logits_new = torch.cat([pred[0] for pred in test_predictions_new])
        test_labels_new = torch.cat([pred[1] for pred in test_predictions_new])
        test_decisions_new = test_logits_new.argmax(-1)
        test_acc_new = torch.mean((test_decisions_new == test_labels_new).float())

        decision_flips = torch.sum(test_decisions != test_decisions_new)
        results.append({
            'num_new_samples': num_new,
            'decision_flips': decision_flips.item(),
            'accuracy': test_acc.item(),
            'accuracy_new': test_acc_new.item(),
        })

    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp_name)
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    for result_dict in results:
        mlflow.log_metrics(result_dict, step=result_dict['num_new_samples'])
    mlflow.end_run()


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
        for batch in dataloader:
            features.append(model(batch[0].to(device)).to('cpu'))
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
