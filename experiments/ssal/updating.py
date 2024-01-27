import os
import copy
import time
import hydra
import torch
import mlflow

from lightning import Trainer
from omegaconf import OmegaConf
from dal_toolbox.datasets import CIFAR10, CIFAR100, SVHN
from dal_toolbox.datasets.utils import PlainTransforms
from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.sngp import RandomFeatureGaussianProcess, SNGPModel
from dal_toolbox.models.laplace import LaplaceLayer, LaplaceModel
from dal_toolbox.metrics import Accuracy, AdaptiveCalibrationError, OODAUROC, OODAUPR, entropy_from_logits
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger
from torch.utils.data import DataLoader, Subset
from utils import DinoFeatureDataset, flatten_cfg


@hydra.main(version_base=None, config_path="./configs", config_name="updating")
def main(args):
    seed_everything(42)  # seed for val split being identical each time

    # First fixed seed for datasets to be identical
    mlflow.set_tracking_uri(uri="file://{}".format(os.path.abspath(args.mlflow_dir)))
    mlflow.set_experiment("Bayesian Updating")
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
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size, shuffle=False)
    ood_loader = DataLoader(ood_ds, batch_size=args.model.predict_batch_size, shuffle=False)

    seed_everything(args.random_seed)
    init_model = build_model(args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)

    # Define indices for training, updating and retraining
    rnd_indices = torch.randperm(len(train_ds))
    train_indices = rnd_indices[:args.num_init_samples]
    new_indices = rnd_indices[args.num_init_samples:args.num_init_samples+args.num_new_samples]
    retrain_indices = rnd_indices[:args.num_init_samples+args.num_new_samples]

    # Train
    base_model = copy.deepcopy(init_model)
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
    trainer.fit(base_model, train_dataloaders=train_loader)

    predictions_base = trainer.predict(base_model, test_loader)
    ood_predictions_base = trainer.predict(base_model, ood_loader)
    test_stats_base = evaluate(predictions_base, ood_predictions_base)
    y_pred_original = torch.cat([pred[0] for pred in predictions_base]).argmax(-1)

    # Updating
    update_model = copy.deepcopy(base_model)
    update_loader = DataLoader(Subset(train_ds, indices=new_indices), batch_size=args.model.train_batch_size,)
    start_time = time.time()
    update_model.update_posterior(update_loader, lmb=args.update_lmb, gamma=args.update_gamma)
    updating_time = time.time() - start_time

    predictions_updated = trainer.predict(update_model, test_loader)
    ood_predictions_updated = trainer.predict(update_model, ood_loader)
    test_stats_updating = evaluate(predictions_updated, ood_predictions_updated)
    test_stats_updating['updating_time'] = updating_time

    y_pred_updated = torch.cat([pred[0] for pred in predictions_updated]).argmax(-1)
    test_stats_updating['decision_flips'] = torch.sum(y_pred_original != y_pred_updated).item()

    # Retraining
    retrain_model = copy.deepcopy(init_model)
    train_loader = DataLoader(
        Subset(train_ds, indices=retrain_indices),
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
    start_time = time.time()
    trainer.fit(retrain_model, train_dataloaders=train_loader)
    retraining_time = time.time() - start_time

    predictions_retrained = trainer.predict(retrain_model, test_loader)
    ood_predictions_retrained = trainer.predict(retrain_model, ood_loader)
    test_stats_retraining = evaluate(predictions_retrained, ood_predictions_retrained)
    test_stats_retraining['retraining_time'] = retraining_time

    y_pred_retrained = torch.cat([pred[0] for pred in predictions_retrained]).argmax(-1)
    test_stats_retraining['decision_flips'] = torch.sum(y_pred_original != y_pred_retrained).item()

    print('Base model:', test_stats_base)
    print('Updated model:', test_stats_updating)
    print('Retrained model:', test_stats_retraining)

    mlflow.log_metrics({f'base_{k}': v for k, v in test_stats_base.items()})
    mlflow.log_metrics({f'updated_{k}': v for k, v in test_stats_updating.items()})
    mlflow.log_metrics({f'retrained_{k}': v for k, v in test_stats_retraining.items()})
    mlflow.end_run()


def evaluate(predictions, ood_predictions):
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])
    test_entropies = entropy_from_logits(test_logits)

    ood_logits = torch.cat([pred[0] for pred in ood_predictions])
    ood_entropies = entropy_from_logits(ood_logits)

    test_stats = {
        'accuracy': Accuracy()(test_logits, test_labels).item(),
        'ACE': AdaptiveCalibrationError()(test_logits, test_labels).item(),
        'AUROC': OODAUROC()(test_entropies, ood_entropies).item(),
        'AUPR': OODAUPR()(test_entropies, ood_entropies).item(),
    }
    return test_stats


def build_dino_model(args):
    dino_model = torch.hub.load('facebookresearch/dinov2', args.dino_model_name)
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
    elif args.model.name == 'laplace':
        model = LaplaceLayer(num_features, num_classes, mean_field_factor=args.model.mean_field_factor)
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
    else:
        raise NotImplementedError()
    return model


if __name__ == '__main__':
    main()
