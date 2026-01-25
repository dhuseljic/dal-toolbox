import time
import hydra
import mlflow
import logging
import warnings

import torch
import torch.nn as nn

import timm
from lightning import Trainer
from omegaconf import OmegaConf

from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning import strategies
from dal_toolbox.active_learning import oracles
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything


from utils import build_image_data, flatten_cfg, build_model
from strategies import Refine, SelectAL, TCM, TAILOR, AutoAL

mlflow.config.enable_async_logging()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="lightning")
# os.environ["POSSIBLE_USER_WARNINGS"] = "off"


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning_finetune")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    data = build_image_data(args)
    train_ds = data.train_dataset
    test_ds = data.test_dataset

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
    al_datamodule.random_init(n_samples=args.dataset.num_init)  # TODO implement in strategy

    model = timm.create_model('resnet18.a1_in1k', pretrained=True)
    model.fc = nn.Linear(512, 10)

    al_datamodule.train_dataloader()  # to init dataset transforms
    batch = next(iter(al_datamodule.train_dataloader()))


    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        barebones=True,
        callbacks=[MetricLogger()],
    )
    al_strategy = build_al_strategy(args, num_classes=10, num_features=512)

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

        strategy_stats = get_strategy_stats(args, al_strategy, i_acq)
        test_stats.update(strategy_stats)
        strategy_artifacts = get_strategy_artifacts(args, al_strategy, i_acq)
        artifacts.update(strategy_artifacts)

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
    from sklearn.metrics import balanced_accuracy_score
    test_logits = torch.cat([pred[0] for pred in predictions])
    test_labels = torch.cat([pred[1] for pred in predictions])

    test_stats = {
        'accuracy': metrics.Accuracy()(test_logits, test_labels).item(),
        'balanced_accuracy': balanced_accuracy_score(test_labels, test_logits.argmax(dim=-1)),
        'NLL': metrics.CrossEntropy()(test_logits, test_labels).item(),
        'BS': metrics.BrierScore()(test_logits, test_labels).item(),
        'ECE': metrics.ExpectedCalibrationError()(test_logits, test_labels).item(),
    }
    return test_stats


def get_strategy_stats(args, al_strategy, i_acq):
    stats = {}
    if i_acq == 0:
        return {}
    if args.al.strategy == 'select_al':
        queried = al_strategy.query_history[-1]
    elif args.al.strategy == 'refine':
        stats = {k: v[-1] for k, v in al_strategy.history.items() if isinstance(v[-1], (int, float))}
    return stats


def get_strategy_artifacts(args, al_strategy, i_acq):
    artifacts = {}
    return artifacts


def build_al_strategy(args, num_classes=None, num_features=None):
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
    elif args.al.strategy == 'refine':
        al_strategy = Refine(
            al_strategies=args.al.refine.strategies,
            progressive_depth=args.al.refine.progressive_depth,
            num_batches=args.al.refine.num_batches,
            alpha=args.al.refine.alpha,
            select_strategy=args.al.refine.select_strategy,
            init_subset_size=args.al.refine.init_subset_size,
            max_pool_size=args.al.refine.max_pool_size,
            filter_acq_size=args.al.refine.filter_acq_size,
            # ERR
            perf_estimation=args.al.aal.perf_estimation,
            look_ahead=args.al.aal.look_ahead,
            num_mc_labels=args.al.aal.num_mc_labels,
            loss=args.al.aal.loss,
            num_retraining_epochs=args.al.aal.num_retraining_epochs,
            eval_gt=args.al.aal.eval_gt,
            device=device
        )
    elif args.al.strategy == 'select_al':
        al_strategy = SelectAL(subset_size=subset_size, device=device)
    elif args.al.strategy == 'tcm':
        al_strategy = TCM(subset_size=subset_size, device=device)
    elif args.al.strategy == 'tailor':
        al_strategy = TAILOR(subset_size=subset_size, device=device)
    elif args.al.strategy == 'autoal':
        al_strategy = AutoAL(args=args, num_classes=num_classes, subset_size=subset_size,
                             feature_dim=num_features, device=device)
    else:
        raise NotImplementedError()
    return al_strategy


if __name__ == '__main__':
    main()
