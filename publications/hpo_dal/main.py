import time
import hydra
import torch
import mlflow
import logging
import warnings

from lightning import Trainer
from omegaconf import OmegaConf

from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning import oracles
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything

from utils import build_datasets, flatten_cfg, build_model

mlflow.config.enable_async_logging()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="lightning")


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    train_ds, test_ds, num_classes = build_datasets(args)
    num_features = len(train_ds[0][0])

    seed_everything(args.random_seed)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    args.dataset.num_init = args.dataset.acq_size if args.dataset.num_init is None else args.dataset.num_init
    al_datamodule.random_init(n_samples=args.dataset.num_init)

    model = build_model(args, num_features=num_features, num_classes=num_classes)
    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        barebones=True,
        callbacks=[MetricLogger()],
    )
    al_strategy = build_al_strategy(args)

    al_history = []
    artifacts_history = []
    for i_acq in range(0, args.dataset.num_acq+1):
        if i_acq != 0:
            stime = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.dataset.acq_size,
            )
            etime = time.time()
            al_datamodule.update_annotations(indices)

        artifacts = {
            'query_indices': indices if i_acq != 0 else al_datamodule.labeled_indices,
        }

        model.reset_states()
        trainer = Trainer(**lightning_trainer_config)
        trainer.fit(model, train_dataloaders=al_datamodule.train_dataloader())
        model.eval()
        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0

        print(f'Cycle {i_acq}:', {k: round(v, 3) for k, v in test_stats.items()}, flush=True)
        al_history.append(test_stats)
        artifacts_history.append(artifacts)

    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(al_history):
        if 'selected_strat' in test_stats:
            test_stats.pop('selected_strat')
        mlflow.log_metrics(test_stats, step=i_acq)
        mlflow.log_dict(artifacts_history[i_acq], f'artifacts_cycle{i_acq:02d}')
    mlflow.end_run()


def evaluate(predictions):
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_labels).item(),
        'NLL': metrics.CrossEntropy()(test_logits, test_labels).item(),
        'BS': metrics.BrierScore()(test_logits, test_labels).item(),
        'ECE': metrics.ExpectedCalibrationError()(test_logits, test_labels).item(),
    }
    return test_stats


def build_al_strategy(args):
    subset_size = args.dataset.subset_size
    device = args.al.device
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'oracle':
        al_strategy = oracles.BoSS()
    elif args.al.strategy == 'margin':
        al_strategy = strategies.MarginSampling(subset_size=subset_size, device=device)
    elif args.al.strategy == 'coreset':
        al_strategy = strategies.CoreSet(subset_size=subset_size, device=device)
    elif args.al.strategy == 'typiclust':
        al_strategy = strategies.TypiClust(subset_size=subset_size, device=device)
    elif args.al.strategy == 'dropquery':
        al_strategy = strategies.DropQuery(subset_size=subset_size, device=device)
    elif args.al.strategy == 'alfamix':
        al_strategy = strategies.AlfaMix(subset_size=subset_size, device=device)
    elif args.al.strategy == 'badge':
        al_strategy = strategies.Badge(subset_size=subset_size, device=device)
    elif args.al.strategy == 'max_herding':
        al_strategy = strategies.MaxHerding(subset_size=subset_size, device=device)
    elif args.al.strategy == 'uncertainty_herding':
        al_strategy = strategies.UncertaintyHerding(subset_size=subset_size, device=device)
    elif args.al.strategy == 'bait':
        al_strategy = strategies.BaitSampling(
            subset_size=subset_size, grad_likelihood='binary_cross_entropy', device=device)
    else:
        raise NotImplementedError()
    return al_strategy


if __name__ == '__main__':
    main()