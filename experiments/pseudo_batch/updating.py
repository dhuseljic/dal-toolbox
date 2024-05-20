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
from rich.progress import track


@hydra.main(version_base=None, config_path="./configs", config_name="updating")
def main(args):
    seed_everything(42)  # seed for val split being identical each time
    print(OmegaConf.to_yaml(args))

    # Setup Data
    train_ds, test_ds, num_classes = build_datasets(
        args, val_split=args.use_val_split, cache_features=args.cache_features)
    test_loader = DataLoader(test_ds, batch_size=args.model.predict_batch_size,
                             shuffle=False, num_workers=args.num_workers)

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
        num_workers=args.num_workers
    )
    trainer = Trainer(**lightning_trainer_config)
    trainer.fit(base_model, train_dataloaders=train_loader)

    # Evaluate
    predictions_base = trainer.predict(base_model, test_loader)
    test_stats_base = evaluate(predictions_base)
    y_pred_original = torch.cat([pred[0] for pred in predictions_base]).argmax(-1)

    update_types = ['first_order', 'second_order', 'mc']
    results = {}
    for num_new in args.num_new_samples:
        update_ds = Subset(train_ds, indices=new_indices[:num_new])
        update_loader = DataLoader(update_ds, batch_size=args.model.train_batch_size)

        if 'second_order' in update_types:
            update_model = copy.deepcopy(base_model)
            start_time = time.time()
            update_model.update_posterior(
                update_loader,
                lmb=args.update_lmb,
                gamma=args.update_gamma,
                cov_likelihood=args.likelihood,
                update_type='second_order'
            )
            updating_time = time.time() - start_time

            predictions_updated = trainer.predict(update_model, test_loader)
            y_pred_updated = torch.cat([pred[0] for pred in predictions_updated]).argmax(-1)
            test_stats_updating = evaluate(predictions_updated)
            test_stats_updating['decision_flips'] = torch.sum(y_pred_original != y_pred_updated).item()
            test_stats_updating['time'] = updating_time

        if 'first_order' in update_types:
            update_model = copy.deepcopy(base_model)
            start_time = time.time()
            update_model.update_posterior(
                update_loader,
                lmb=args.update_lmb,
                gamma=args.update_gamma_fo,
                cov_likelihood=args.likelihood,
                update_type='first_order'
            )
            updating_time = time.time() - start_time

            predictions_updated = trainer.predict(update_model, test_loader)
            y_pred_updated = torch.cat([pred[0] for pred in predictions_updated]).argmax(-1)
            test_stats_updating_fo = evaluate(predictions_updated)
            test_stats_updating_fo['decision_flips'] = torch.sum(y_pred_original != y_pred_updated).item()
            test_stats_updating_fo['time'] = updating_time

        if 'mc' in update_types:
            update_model_mc = copy.deepcopy(base_model)
            start_time = time.time()
            sampled_params, weights = update_mc(
                update_model_mc,
                update_loader,
                mc_samples=args.model.mc_samples,
                gamma=args.update_gamma_mc
            )
            updating_time = time.time() - start_time
            predictions_updated_mc = predict_from_mc(
                update_model_mc, test_loader, sampled_params=sampled_params, weights=weights)
            test_stats_updating_mc = evaluate(predictions_updated_mc)
            y_pred_updated_mc = torch.cat([pred[0] for pred in predictions_updated_mc]).argmax(-1)
            test_stats_updating_mc['decision_flips'] = torch.sum(y_pred_original != y_pred_updated_mc).item()
            test_stats_updating_mc['time'] = updating_time

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
        print('Baseline model:\t\t', test_stats_base)
        print('FO Updated model:\t', test_stats_updating_fo)
        print('MC Updated model:\t', test_stats_updating_mc)
        print('SO Updated model:\t', test_stats_updating)
        print('Retrained model:\t', test_stats_retraining)
        results[num_new] = {
            'base': test_stats_base,
            'updated': test_stats_updating,
            'fo_updated': test_stats_updating_fo,
            'mc_updated': test_stats_updating_mc,
            'retrained': test_stats_retraining
        }

    # Logging
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    for num_new in results:
        res_dict = results[num_new]
        test_stats_base = res_dict['base']
        test_stats_updating = res_dict['updated']
        test_stats_updating_mc = res_dict['mc_updated']
        test_stats_updating_fo = res_dict['fo_updated']
        test_stats_retraining = res_dict['retrained']

        mlflow.log_metrics({f'base_{k}': v for k, v in test_stats_base.items()}, step=num_new)
        mlflow.log_metrics({f'fo_updated_{k}': v for k, v in test_stats_updating_fo.items()}, step=num_new)
        mlflow.log_metrics({f'mc_updated_{k}': v for k, v in test_stats_updating_mc.items()}, step=num_new)
        mlflow.log_metrics({f'updated_{k}': v for k, v in test_stats_updating.items()}, step=num_new)
        mlflow.log_metrics({f'retrained_{k}': v for k, v in test_stats_retraining.items()}, step=num_new)

        print('Number of new samples:', num_new)
        print('Baseline model:\t\t', test_stats_base)
        print('FO Updated model:\t', test_stats_updating_fo)
        print('MC Updated model:\t', test_stats_updating_mc)
        print('SO Updated model:\t', test_stats_updating)
        print('Retrained model:\t', test_stats_retraining)
        print('=' * 20)
    mlflow.end_run()


def evaluate(predictions):
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_labels).item(),
        'ECE': metrics.ExpectedCalibrationError()(test_logits, test_labels).item(),
        'ACE': metrics.AdaptiveCalibrationError()(test_logits, test_labels).item(),
    }
    return test_stats


def update_mc(model, update_loader, mc_samples, gamma=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Get all features
    phis_list = []
    targets_list = []
    for inputs, targets in update_loader:
        with torch.no_grad():
            _, phis = model.model(inputs.to(device), return_features=True)
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
    log_prior = torch.log(torch.ones(n_members, device=device) / n_members)  # uniform prior
    log_likelihood = log_probas_mc.permute(1, 0, 2)[:, range(n_samples), targets].sum(dim=1)
    log_posterior = log_prior + gamma*log_likelihood
    weights = torch.softmax(log_posterior, dim=0)

    return sampled_params.cpu(), weights.cpu()


@torch.no_grad()
def predict_from_mc(model, dataloader, sampled_params, weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampled_params = sampled_params.to(device)
    model = model.to(device)
    predictions = []
    for batch in track(dataloader, description='Predicting with MC '):
        inputs = batch[0].to(device)
        targets = batch[1]

        with torch.no_grad():
            _, phis = model.model(inputs, return_features=True)

        logits_mc = torch.einsum('nd,ekd->nek', phis, sampled_params)
        probas_mc = logits_mc.softmax(-1)

        if weights is not None:
            weights = weights.to(probas_mc)
            probas = torch.einsum('e,nek->nk', weights, probas_mc)
        else:
            probas = torch.mean(probas_mc, dim=1)
        logits = probas.log()
        predictions.append([logits.cpu(), targets])

    return predictions


if __name__ == '__main__':
    main()

