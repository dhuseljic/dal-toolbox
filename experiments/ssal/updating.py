import os
import copy
import time
import hydra
import torch
import mlflow

from lightning import Trainer
from omegaconf import OmegaConf
from dal_toolbox import metrics
from dal_toolbox.utils import seed_everything
from dal_toolbox.models.utils.callbacks import MetricLogger
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import pairwise_distances
from utils import DinoFeatureDataset, flatten_cfg, build_data, build_model, build_ood_data, build_dino_model


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
    test_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.test_dataset,
                                 normalize_features=True, cache=True)
    ood_ds = DinoFeatureDataset(dino_model=dino_model, dataset=ood_data.test_dataset,
                                normalize_features=True, cache=True)
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size, shuffle=False)

    seed_everything(args.random_seed)
    init_model = build_model(
        args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)

    # Define indices for training, updating and retraining
    rnd_indices = torch.randperm(len(train_ds))
    train_indices = rnd_indices[:args.num_init_samples]
    new_indices = rnd_indices[args.num_init_samples:args.num_init_samples+args.num_new_samples]
    retrain_indices = rnd_indices[:args.num_init_samples+args.num_new_samples]

    # OOD eval datasets
    num_samples_per_init = args.num_ood_samples // args.num_init_samples
    dist = pairwise_distances(train_ds.features[train_indices], train_ds.features)
    id_indices = dist.argsort(axis=-1)[:, 1:num_samples_per_init+1].flatten()
    ood_indices = range(len(id_indices))
    id_loader = DataLoader(train_ds, sampler=id_indices, batch_size=args.model.train_batch_size)
    ood_loader = DataLoader(ood_ds, sampler=ood_indices, batch_size=args.model.train_batch_size)

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

    # Evaluate
    predictions_base = trainer.predict(base_model, test_loader)
    test_stats_base = evaluate(predictions_base)
    y_pred_original = torch.cat([pred[0] for pred in predictions_base]).argmax(-1)

    id_predictions = trainer.predict(base_model, id_loader)
    ood_predictions = trainer.predict(base_model, ood_loader)
    ood_stats = evaluate_ood(id_predictions, ood_predictions)
    test_stats_base.update(ood_stats)

    if args.model.name == 'deterministic':
        print('Base model:', test_stats_base)
        mlflow.log_metrics({f'base_{k}': v for k, v in test_stats_base.items()})
        mlflow.end_run()
        return

    # Updating
    update_model = copy.deepcopy(base_model)
    update_loader = DataLoader(Subset(train_ds, indices=new_indices), batch_size=args.model.train_batch_size,)
    start_time = time.time()
    update_model.update_posterior(update_loader, lmb=args.update_lmb,
                                  gamma=args.update_gamma, likelihood=args.likelihood)
    updating_time = time.time() - start_time

    predictions_updated = trainer.predict(update_model, test_loader)
    test_stats_updating = evaluate(predictions_updated)

    id_predictions = trainer.predict(update_model, id_loader)
    ood_predictions = trainer.predict(update_model, ood_loader)
    ood_stats = evaluate_ood(id_predictions, ood_predictions)
    test_stats_updating.update(ood_stats)

    y_pred_updated = torch.cat([pred[0] for pred in predictions_updated]).argmax(-1)
    test_stats_updating['decision_flips'] = torch.sum(y_pred_original != y_pred_updated).item()
    test_stats_updating['updating_time'] = updating_time

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
    test_stats_retraining = evaluate(predictions_retrained)

    id_predictions = trainer.predict(retrain_model, id_loader)
    ood_predictions = trainer.predict(retrain_model, ood_loader)
    ood_stats = evaluate_ood(id_predictions, ood_predictions)
    test_stats_retraining.update(ood_stats)

    y_pred_retrained = torch.cat([pred[0] for pred in predictions_retrained]).argmax(-1)
    test_stats_retraining['decision_flips'] = torch.sum(y_pred_original != y_pred_retrained).item()
    test_stats_retraining['retraining_time'] = retraining_time

    print('Base model:', test_stats_base)
    print('Updated model:', test_stats_updating)
    print('Retrained model:', test_stats_retraining)

    mlflow.log_metrics({f'base_{k}': v for k, v in test_stats_base.items()})
    mlflow.log_metrics({f'updated_{k}': v for k, v in test_stats_updating.items()})
    mlflow.log_metrics({f'retrained_{k}': v for k, v in test_stats_retraining.items()})
    mlflow.end_run()


def evaluate(predictions):
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_labels).item(),
        'NLL': metrics.CrossEntropy()(test_logits, test_labels).item(),
        'BS': metrics.BrierScore()(test_logits, test_labels).item(),
        'ECE': metrics.ExpectedCalibrationError()(test_logits, test_labels).item(),
        'ACE': metrics.AdaptiveCalibrationError()(test_logits, test_labels).item(),
        'reliability': metrics.BrierScoreDecomposition()(test_logits, test_labels)['reliability']
    }
    return test_stats


def evaluate_ood(id_predictions, ood_predictions):
    id_entropy = metrics.entropy_from_logits(torch.cat([pred[0] for pred in id_predictions]))
    ood_entropy = metrics.entropy_from_logits(torch.cat([pred[0] for pred in ood_predictions]))

    ood_stats = {
        'AUROC': metrics.OODAUROC()(id_entropy, ood_entropy).item(),
        'AUPR': metrics.OODAUPR()(id_entropy, ood_entropy).item()
    }
    return ood_stats


if __name__ == '__main__':
    main()
