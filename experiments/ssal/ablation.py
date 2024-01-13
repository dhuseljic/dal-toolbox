import hydra
import torch
import mlflow
from lightning import Trainer
from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import PlainTransforms
from dal_toolbox.models.sngp import RandomFeatureGaussianProcess, SNGPModel
from dal_toolbox.metrics import Accuracy
from dal_toolbox.utils import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from omegaconf import OmegaConf



@hydra.main(version_base=None, config_path="./configs", config_name="ablation")
def main(args):
    seed_everything(args.random_seed)

    mlflow.set_experiment("Ablation")
    mlflow.start_run()
    mlflow.log_params(dict(args))

    # Setup
    dino_model = build_dino_model(args)
    data = build_data(args)
    train_ds = DinoFeatureDataset(
        dino_model=dino_model, dataset=data.train_dataset, normalize_features=True)
    test_ds = DinoFeatureDataset(
        dino_model=dino_model, dataset=data.test_dataset, normalize_features=True)

    model = build_model(
        args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)

    # Train
    train_indices = torch.randperm(len(train_ds))[:args.num_train_samples]
    train_loader = DataLoader(
        Subset(train_ds, indices=train_indices),
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=len(train_indices) > args.train_batch_size,
    )
    trainer = Trainer(
        max_epochs=args.num_epochs
    )
    trainer.fit(model, train_dataloaders=train_loader)

    # Eval
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    predictions = trainer.predict(model, test_loader)

    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])
    test_stats = {
        'accuracy': Accuracy()(test_logits, test_labels).item(),
    }

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
    else:
        raise NotImplementedError()
    return data


def build_model(args, **kwargs):
    num_features = kwargs['num_features']
    num_classes = kwargs['num_classes']
    model = RandomFeatureGaussianProcess(
        in_features=num_features,
        out_features=num_classes,
        num_inducing=args.num_inducing,
        kernel_scale=args.kernel_scale,
        scale_random_features=args.scale_random_features,
        optimize_kernel_scale=args.optimize_kernel_scale,
        mean_field_factor=args.mean_field_factor,
    )
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs)

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

    def __init__(self, dino_model, dataset, normalize_features=True, device='cuda'):
        features, labels = self.get_dino_features(dino_model, dataset, device)

        if normalize_features:
            features_mean = features.mean(0)
            features_std = features.std(0) + 1e-9
            features = (features - features_mean) / features_std

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def get_dino_features(self, dino_model, dataset, device):
        print('Getting dino features..')
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)

        features = []
        labels = []
        dino_model.to(device)
        for batch in tqdm(dataloader):
            features.append(dino_model(batch[0].to(device)).to('cpu'))
            labels.append(batch[-1])
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels


if __name__ == '__main__':
    main()
