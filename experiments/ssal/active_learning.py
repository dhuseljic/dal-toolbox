import os
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
from utils import DinoFeatureDataset, flatten_cfg, build_data, build_model, build_dino_model


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    seed_everything(42)
    print(OmegaConf.to_yaml(args))

    dino_model = build_dino_model(args)
    data = build_data(args)

    train_ds = DinoFeatureDataset(dino_model, dataset=data.train_dataset,
                                  cache=True, cache_dir=args.dino_cache_dir)
    test_ds = DinoFeatureDataset(dino_model, dataset=data.val_dataset,
                                 cache=True, cache_dir=args.dino_cache_dir)
    # TODO: use test_ds only at the end
    # test_ds = DinoFeatureDataset(dino_model, dataset=data.test_dataset, cache=True)

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
    model = build_model(args, num_features=dino_model.norm.normalized_shape[0], num_classes=data.num_classes)
    for i_acq in range(0, args.al.num_acq+1):
        if i_acq != 0:
            print('Querying..')
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
        print(f'Cycle {i_acq}:', test_stats, flush=True)
        history.append(test_stats)

    mlflow.set_tracking_uri(uri="file://{}".format(os.path.abspath(args.mlflow_dir)))
    mlflow.set_experiment("Active Learning")
    mlflow.start_run()
    mlflow.log_params(flatten_cfg(args))
    for i_acq, test_stats in enumerate(history):
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
        al_strategy = strategies.RandomSampling()
    elif args.al.strategy == 'margin':
        al_strategy = strategies.MarginSampling(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_margin':
        strat = strategies.MarginSampling(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.al.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'badge':
        al_strategy = strategies.Badge(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_badge':
        strat = strategies.Badge(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.al.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'batch_bald':
        al_strategy = strategies.BatchBALDSampling(subset_size=args.al.subset_size)
    elif args.al.strategy == 'pseudo_bald':
        strat = strategies.BALDSampling(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.al.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'typiclust':
        al_strategy = strategies.TypiClust(subset_size=args.al.subset_size)
    elif args.al.strategy == 'bait':
        al_strategy = strategies.BaitSampling(
            subset_size=args.al.subset_size, fisher_approx=args.al.fisher_approximation)
    elif args.al.strategy == 'pseudo_bait':
        strat = strategies.BaitSampling(subset_size=args.al.subset_size,
                                        fisher_approx=args.al.fisher_approximation)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.al.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
    elif args.al.strategy == 'decision_flip':
        strat = DecisionFlipSampling(subset_size=args.al.subset_size)
        al_strategy = PseudoBatch(al_strategy=strat, update_every=args.al.update_every,
                                  gamma=args.update_gamma, subset_size=args.al.subset_size)
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
        # Copy data module to avoid querying the same instances
        al_datamodule_batch = copy.deepcopy(al_datamodule)
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)

        if acq_size % self.update_every != 0:
            raise ValueError('Acquisition size must be divisible by `update_every`.')

        indices = []
        from tqdm.auto import tqdm
        for _ in tqdm(range(acq_size // self.update_every)):
            # Sample via simple strategy
            idx = self.al_strategy.query(
                model=model,
                al_datamodule=al_datamodule_batch,
                acq_size=self.update_every
            )

            # Get the element and label from the dataloader
            data = unlabeled_dataloader.dataset[idx]
            sample = data[0]
            target = data[1]

            # Update the model
            model.cpu()
            model.update_posterior(zip([sample], [target]), gamma=self.gamma, lmb=self.lmb)
            al_datamodule_batch.update_annotations(idx)
            indices.extend(idx)
        actual_indices = indices
        return actual_indices


class DecisionFlipSampling(strategies.Query):
    def __init__(self, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size
        self.gamma = 10
        self.lmb = 1

    @torch.no_grad()
    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)

        unlabeled_inputs = []
        for batch in unlabeled_dataloader:
            inputs = batch[0]
            unlabeled_inputs.append(inputs)
        unlabeled_inputs = torch.cat(unlabeled_inputs)
        unlabeled_logits = model.model.forward_mean_field(unlabeled_inputs)
        y_pred = unlabeled_logits.argmax(-1)

        _, num_classes = unlabeled_logits.shape

        # Number of decision flips when the label of a sample would be different the prognosed
        from tqdm import tqdm
        i_sample = 0
        scores = torch.zeros(len(unlabeled_indices))
        for batch in tqdm(unlabeled_dataloader):
            for sample in batch[0]:
                for i_cls in range(num_classes):
                    updated_model = copy.deepcopy(model)
                    iterator = zip([sample.view(1, -1)], [torch.tensor(i_cls).view(-1)])
                    updated_model.update_posterior(iterator, lmb=self.lmb, gamma=self.gamma)
                    updated_logits = updated_model.model.forward_mean_field(unlabeled_inputs)
                    updated_pred = updated_logits.argmax(-1)
                    # scores[i_sample] += unlabeled_logits[i_sample].softmax(-1)[i_cls] * torch.sum(y_pred != updated_pred).item()
                    scores[i_sample] += torch.sum(y_pred != updated_pred).item()
                i_sample += 1

        _, indices = scores.topk(acq_size)
        actual_indices = [unlabeled_indices[i] for i in indices]
        if return_utilities:
            return actual_indices, scores
        return actual_indices


if __name__ == '__main__':
    main()
