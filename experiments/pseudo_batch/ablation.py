import os
import hydra
import torch
import mlflow

from lightning import Trainer
from omegaconf import OmegaConf
from dal_toolbox.datasets import CIFAR10, CIFAR100, SVHN
from dal_toolbox.datasets.utils import PlainTransforms
from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.sngp import RandomFeatureGaussianProcess, SNGPModel
from dal_toolbox.metrics import Accuracy, AdaptiveCalibrationError, OODAUROC, OODAUPR, entropy_from_logits
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger
from torch.utils.data import DataLoader, Subset
from utils import DinoFeatureDataset, flatten_cfg


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
        max_epochs=args.model.num_epochs,
        default_root_dir=args.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[MetricLogger()],
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
    if args.model.name == 'sngp':
        model = RandomFeatureGaussianProcess(
            in_features=num_features,
            out_features=num_classes,
            num_inducing=args.model.num_inducing,
            kernel_scale=args.model.kernel_scale,
            scale_random_features=args.model.scale_random_features,
            optimize_kernel_scale=args.model.optimize_kernel_scale,
            mean_field_factor=args.model.mean_field_factor,
        )
    elif args.model.name == 'linear':
        model = torch.nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError()

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

    if args.model.name == 'sngp':
        model = SNGPModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    elif args.model.name == 'linear':
        model = DeterministicModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    else:
        raise NotImplementedError()

    return model


if __name__ == '__main__':
    main()
