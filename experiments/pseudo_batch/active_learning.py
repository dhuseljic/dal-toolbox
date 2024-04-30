import os
import time
import hydra
import torch
import mlflow
import copy

from omegaconf import OmegaConf
from lightning import Trainer

from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning import strategies
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything
from utils import build_datasets, flatten_cfg, build_model
from torch.utils.data import DataLoader, Subset


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    train_ds, test_ds, num_classes = build_datasets(
        args, val_split=args.use_val_split, cache_features=args.cache_features)

    seed_everything(args.random_seed)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    if args.al.init_method == 'random':
        al_datamodule.random_init(n_samples=args.al.num_init_samples)
    elif args.al.init_method == 'diverse_dense':
        al_datamodule.diverse_dense_init(n_samples=args.al.num_init_samples)
    else:
        raise NotImplementedError()
    al_strategy = build_al_strategy(args)

    history = []
    num_features = 384  # len(train_ds[0][0])
    model = build_model(args, num_features=num_features, num_classes=num_classes)
    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        default_root_dir=args.output_dir,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[MetricLogger()],
    )
    for i_acq in range(0, args.al.num_acq+1):
        if i_acq != 0:
            print('Querying..')
            stime = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al.acq_size,
            )
            etime = time.time()
            al_datamodule.update_annotations(indices)

        model.reset_states()
        trainer = Trainer(**lightning_trainer_config)
        trainer.fit(model, train_dataloaders=al_datamodule)

        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        test_stats['query_time'] = etime - stime if i_acq != 0 else 0
        print(f'Cycle {i_acq}:', test_stats, flush=True)
        history.append(test_stats)

    mlflow.set_tracking_uri(uri="{}".format(args.mlflow_uri))
    mlflow.set_experiment("Active Learning")
    mlflow.start_run()
    log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(history):
        log_metrics(test_stats, step=i_acq)
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


def build_al_strategy(args):
    if args.al.strategy == 'random':
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'margin':
        al_strategy = strategies.MarginSampling(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_margin':
        strat = strategies.MarginSampling(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'badge':
        al_strategy = strategies.Badge(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_badge':
        strat = strategies.Badge(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'batch_bald':
        al_strategy = strategies.BatchBALDSampling(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_bald':
        strat = strategies.BALDSampling(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'bait':
        al_strategy = strategies.BaitSampling(
            subset_size=args.al.subset_size,
            grad_likelihood=args.al.bait.grad_likelihood,
            device=args.al.device
        )
    elif args.al.strategy == 'pseudo_bait':
        strat = strategies.BaitSampling(
            subset_size=args.al.subset_size,
            grad_likelihood=args.al.bait.grad_likelihood,
            device=args.al.device
        )
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'typiclust':
        al_strategy = strategies.TypiClust(subset_size=args.al.subset_size)
    else:
        raise NotImplementedError()
    return al_strategy


class PseudoBatch(strategies.Query):
    def __init__(self, al_strategy, update_every=1, gamma=10, lmb=1, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.al_strategy = al_strategy
        self.gamma = gamma
        self.lmb = lmb
        self.update_every = update_every

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        dataset = unlabeled_dataloader.dataset

        if acq_size % self.update_every != 0:
            raise ValueError('Acquisition size must be divisible by `update_every`.')

        indices = []
        from rich.progress import track
        for _ in track(range(acq_size // self.update_every), 'PseudoBatch: Querying'):
            # Copy data module to avoid querying the same instances
            al_datamodule_batch = copy.deepcopy(al_datamodule)
            al_datamodule_batch.update_annotations(indices)

            # Sample via simple strategy
            idx = self.al_strategy.query(
                model=model,
                al_datamodule=al_datamodule_batch,
                acq_size=self.update_every
            )

            # Get the element and label from the dataloader
            data = dataset[idx]
            sample = data[0]
            target = data[1]

            # Update the model
            model.cpu()
            model.update_posterior(zip([sample], [target]), gamma=self.gamma, lmb=self.lmb)
            indices.extend(idx)
        actual_indices = indices
        return actual_indices


def log_params(*args, **kwargs):
    retry = 5
    for _ in range(retry):
        try:
            mlflow.log_params(*args, **kwargs)
            return
        except:
            time.sleep(1)


def log_metrics(*args, **kwargs):
    retry = 5
    for _ in range(retry):
        try:
            mlflow.log_metrics(*args, **kwargs)
            return
        except:
            time.sleep(1)


if __name__ == '__main__':
    main()
