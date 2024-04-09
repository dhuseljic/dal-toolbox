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
from utils import flatten_cfg, build_datasets, build_model


@hydra.main(version_base=None, config_path="./configs", config_name="updating")
def main(args):
    seed_everything(42)  # seed for val split being identical each time
    print(OmegaConf.to_yaml(args))

    # Setup Data
    train_ds, test_ds, num_classes = build_datasets(args)
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size, shuffle=False)

    seed_everything(args.random_seed)
    num_features = len(train_ds[0][0])
    init_model = build_model(args, num_features=num_features, num_classes=num_classes)

    # Define indices for training, updating and retraining
    rnd_indices = torch.randperm(len(train_ds))
    base_indices = rnd_indices[:args.num_init_samples]
    new_indices = rnd_indices[args.num_init_samples:args.num_init_samples+max(args.num_new_samples)]
    retrain_indices = rnd_indices[:args.num_init_samples+max(args.num_new_samples)]

    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        default_root_dir=args.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[MetricLogger()],
    )

    # Train
    base_model = copy.deepcopy(init_model)
    train_loader = DataLoader(
        Subset(train_ds, indices=base_indices),
        batch_size=args.model.train_batch_size,
        shuffle=True,
        drop_last=len(base_indices) > args.model.train_batch_size,
    )
    trainer = Trainer(**lightning_trainer_config)
    trainer.fit(base_model, train_dataloaders=train_loader)

    # Evaluate
    predictions_base = trainer.predict(base_model, test_loader)
    test_stats_base = evaluate(predictions_base)
    y_pred_original = torch.cat([pred[0] for pred in predictions_base]).argmax(-1)

    results = {}
    for num_new in args.num_new_samples:
        # Second-order updating
        update_model = copy.deepcopy(base_model)
        update_ds = Subset(train_ds, indices=new_indices[:num_new])
        update_loader = DataLoader(update_ds, batch_size=args.model.train_batch_size)
        start_time = time.time()
        update_model.update_posterior(update_loader, lmb=args.update_lmb,
                                      gamma=args.update_gamma, likelihood=args.likelihood)
        updating_time = time.time() - start_time

        predictions_updated = trainer.predict(update_model, test_loader)
        test_stats_updating = evaluate(predictions_updated)
        y_pred_updated = torch.cat([pred[0] for pred in predictions_updated]).argmax(-1)
        test_stats_updating['decision_flips'] = torch.sum(y_pred_original != y_pred_updated).item()
        test_stats_updating['time'] = updating_time

        # Update using Bayes theorem (i.e., Monte Carlo)
        update_model_mc = copy.deepcopy(base_model)
        update_ds_mc = Subset(train_ds, indices=new_indices[:num_new])
        update_loader_mc = DataLoader(update_ds_mc, batch_size=args.model.train_batch_size)
        start_time = time.time()
        sampled_params, weights = update_mc(
            update_model_mc, update_loader_mc, mc_samples=args.model.mc_samples)
        updating_time = time.time() - start_time
        predictions_updated_mc = predict_from_mc(test_loader, sampled_params=sampled_params, weights=weights)
        test_stats_mc_updating = evaluate(predictions_updated_mc)
        y_pred_updated_mc = torch.cat([pred[0] for pred in predictions_updated_mc]).argmax(-1)
        test_stats_mc_updating['decision_flips'] = torch.sum(y_pred_original != y_pred_updated_mc).item()
        test_stats_mc_updating['time'] = updating_time

        # Retraining
        retrain_model = copy.deepcopy(init_model)
        retrain_ds = Subset(train_ds, indices=retrain_indices[:args.num_init_samples+num_new])
        retrain_loader = DataLoader(retrain_ds, batch_size=args.model.train_batch_size,
                                    shuffle=True, drop_last=len(base_indices) > args.model.train_batch_size)
        trainer = Trainer(
            max_epochs=args.model.num_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[MetricLogger()],
        )
        start_time = time.time()
        trainer.fit(retrain_model, train_dataloaders=retrain_loader)
        retraining_time = time.time() - start_time

        predictions_retrained = trainer.predict(retrain_model, test_loader)
        test_stats_retraining = evaluate(predictions_retrained)
        y_pred_retrained = torch.cat([pred[0] for pred in predictions_retrained]).argmax(-1)
        test_stats_retraining['decision_flips'] = torch.sum(y_pred_original != y_pred_retrained).item()
        test_stats_retraining['time'] = retraining_time

        print('Number of new samples:', num_new)
        print('Base model:', test_stats_base)
        print('Updated model:', test_stats_updating)
        print('MC Updated model:', test_stats_mc_updating)
        print('Retrained model:', test_stats_retraining)
        print('=' * 20)
        results[num_new] = {
            'base': test_stats_base,
            'updated': test_stats_updating,
            'mc_updated': test_stats_mc_updating,
            'retrained': test_stats_retraining
        }

    # Logging
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    mlflow.set_experiment("Updating")
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    for num_new in results:
        res_dict = results[num_new]
        test_stats_base = res_dict['base']
        test_stats_updating = res_dict['updated']
        test_stats_mc_updating = res_dict['mc_updated']
        test_stats_retraining = res_dict['retrained']

        mlflow.log_metrics({f'base_{k}': v for k, v in test_stats_base.items()}, step=num_new)
        mlflow.log_metrics({f'updated_{k}': v for k, v in test_stats_updating.items()}, step=num_new)
        mlflow.log_metrics({f'mc_updated_{k}': v for k, v in test_stats_mc_updating.items()}, step=num_new)
        mlflow.log_metrics({f'retrained_{k}': v for k, v in test_stats_retraining.items()}, step=num_new)

        print('Number of new samples:', num_new)
        print('Base model:', test_stats_base)
        print('Updated model:', test_stats_updating)
        print('MC Updated model:', test_stats_mc_updating)
        print('Retrained model:', test_stats_retraining)
        print('=' * 20)
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


def update_mc(model, update_loader, mc_samples):
    # Get all features
    phis_list = []
    targets_list = []
    for inputs, targets in update_loader:
        with torch.no_grad():
            _, phis = model.model(inputs, return_features=True)
    phis_list.append(phis)
    targets_list.append(targets)
    phis = torch.cat(phis_list)
    targets = torch.cat(targets_list)

    # Sample hypothesis
    from dal_toolbox.models.laplace import LaplaceLayer
    for m in model.model.modules():
        if isinstance(m, LaplaceLayer):
            laplace_layer = m
    mean = laplace_layer.layer.weight.data
    prec = laplace_layer.precision_matrix.data
    dist = torch.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    sampled_params = dist.sample(sample_shape=(mc_samples,))

    # Get precitions from hypothesis
    logits_mc = torch.einsum('nd,ekd->nek', phis, sampled_params)
    log_probas_mc = logits_mc.log_softmax(-1)

    # Bayes theorem for update
    n_samples, n_members, _ = logits_mc.shape
    log_prior = torch.log(torch.ones(n_members) / n_members)  # uniform prior
    log_likelihood = log_probas_mc.permute(1, 0, 2)[:, range(n_samples), targets].sum(dim=1)
    log_posterior = log_prior + log_likelihood
    weights = torch.softmax(log_posterior, dim=0)

    return sampled_params, weights


@torch.no_grad()
def predict_from_mc(dataloader, sampled_params, weights):
    predictions = []
    corrects = 0
    num_samples = 0
    for batch in dataloader:
        inputs = batch[0]
        targets = batch[1]
        logits_mc = torch.einsum('nd,ekd->nek', inputs, sampled_params)
        probas_mc = logits_mc.softmax(-1)

        corrects += torch.sum((probas_mc.mean(1).argmax(-1) == targets).float()).item()
        num_samples += len(targets)

        if weights is not None:
            probas = torch.einsum('e,nek->nk', weights, probas_mc)
        else:
            probas = torch.mean(probas_mc, dim=1)
        logits = probas.log()
        predictions.append([logits, targets])

    return predictions


if __name__ == '__main__':
    main()
