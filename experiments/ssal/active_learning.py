import os
import hydra
import torch
import mlflow

from omegaconf import OmegaConf
from lightning import Trainer

from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import RandomSampling, EntropySampling, TypiClust, Query
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything
from utils import DinoFeatureDataset, flatten_cfg, build_data, build_model, build_dino_model


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    seed_everything(42)

    mlflow.set_tracking_uri(uri="file://{}".format(os.path.abspath(args.mlflow_dir)))
    mlflow.set_experiment("Active Learning")
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    print(OmegaConf.to_yaml(args))

    dino_model = build_dino_model(args)
    data = build_data(args)

    train_ds = DinoFeatureDataset(dino_model, dataset=data.train_dataset,
                                  normalize_features=True, cache=True, cache_dir=args.dino_cache_dir)
    test_ds = DinoFeatureDataset(dino_model, dataset=data.val_dataset,
                                 normalize_features=True, cache=True, cache_dir=args.dino_cache_dir)
    # TODO: use test_ds only at the end
    # test_ds = DinoFeatureDataset(dino_model, dataset=data.test_dataset, normalize_features=True, cache=True)

    seed_everything(args.random_seed)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size,
    )
    al_datamodule.random_init(n_samples=args.al.num_init_samples)
    al_strategy = build_al_strategy(args)

    model = build_model(args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)
    for i_acq in range(0, args.al.num_acq+1):
        if i_acq != 0:
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al.acq_size,
            )
            al_datamodule.update_annotations(indices)

        model.reset_states()
        trainer = Trainer(
            max_epochs=args.model.num_epochs,
            default_root_dir=args.output_dir,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[MetricLogger()],
        )
        trainer.fit(model, train_dataloaders=al_datamodule)

        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_stats = evaluate(predictions)
        print(f'Cycle {i_acq}:', test_stats)
        mlflow.log_metrics(test_stats, step=i_acq)

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
        al_strategy = RandomSampling()
    elif args.al.strategy == 'entropy':
        al_strategy = EntropySampling()
    elif args.al.strategy == 'typiclust':
        al_strategy = TypiClust()
    elif args.al.strategy == 'pseudo_entropy':
        al_strategy = PseudoBatch(al_strategy=EntropySampling())
    else:
        raise NotImplementedError()
    return al_strategy


class PseudoBatch(Query):
    def __init__(self, al_strategy, gamma=1, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.al_strategy = al_strategy
        self.gamma = gamma

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)

        indices = []
        for _ in range(acq_size):
            # Sample via simple strategy
            idx = self.al_strategy.query(model=model, al_datamodule=al_datamodule, acq_size=1)[0]

            # Get the element and label from the dataloader
            data = unlabeled_dataloader.dataset[idx]
            sample = data[0].view(1, -1)
            target = data[1].view(-1)

            # Update the model
            model.cpu()
            model.update_posterior(zip([sample], [target]), gamma=self.gamma)
            indices.append(idx)
        actual_indices = indices
        return actual_indices


if __name__ == '__main__':
    main()
