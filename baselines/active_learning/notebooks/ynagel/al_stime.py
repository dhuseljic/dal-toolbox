import datetime
import gc
import json
import logging
import os
import sys
import time
from itertools import product

import lightning as L
import torch
from IPython.utils import io
from tqdm import tqdm

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge, typiclust, xpal, xpalclust, \
    randomclust
# noinspection PyUnresolvedReferences
from dal_toolbox.datasets.utils import FeatureDataset, FeatureDatasetWrapper
from dal_toolbox.models import deterministic
from dal_toolbox.utils import seed_everything, kernels, _calculate_mean_gamma


def al_cycle_test(random_seed, strategy, acq_size, data_file, output_dir, subset_size):
    seed_everything(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    # Necessary for logging
    results = {"query_times": []}

    # Setup Dataset
    data = FeatureDatasetWrapper(data_file)
    feature_size = data.num_features

    features = torch.stack([batch[0] for batch in data.train_dataset])

    trainset = data.train_dataset
    queryset = data.query_dataset
    valset = data.val_dataset
    testset = data.test_dataset

    num_classes = data.num_classes

    # Setup Query
    al_strategy = build_al_strategy(name=strategy, subset_size=subset_size, kernel="rbf", alpha=1e-3, precomputed=True,
                                    gamma="calculate", num_classes=num_classes, train_features=features)

    # Setup Model
    accelerator = "cpu"

    model = build_model(num_epochs=10, num_classes=num_classes, feature_size=feature_size)

    # Setup AL Module
    al_datamodule = ActiveLearningDataModule(
        train_dataset=trainset,
        query_dataset=queryset,
        val_dataset=valset,
        test_dataset=testset,
        train_batch_size=64,
        predict_batch_size=256,
    )

    if strategy == "random":
        t1 = time.time()
        al_datamodule.random_init(n_samples=acq_size)
    else:
        init_al_strategy = al_strategy
        t1 = time.time()
        indices = init_al_strategy.query(
            model=model,
            al_datamodule=al_datamodule,
            acq_size=acq_size
        )
        al_datamodule.update_annotations(indices)
    query_eta = datetime.timedelta(seconds=int(time.time() - t1))
    gc.collect()  # init_al_strategy sometimes takes too long to be automatically collected

    # Active Learning Cycles
    for i_acq in range(0, 9 + 1):
        if i_acq != 0:
            t1 = time.time()
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=acq_size
            )
            al_datamodule.update_annotations(indices)
            query_eta = datetime.timedelta(seconds=int(time.time() - t1))

        results["query_times"].append(query_eta.total_seconds())
        #  model cold start
        model.reset_states()

        # Train with updated annotations
        trainer = L.Trainer(
            max_epochs=10,
            enable_checkpointing=False,
            accelerator=accelerator,
            default_root_dir=output_dir,
            enable_progress_bar=False,
            check_val_every_n_epoch=25,
            enable_model_summary=False
        )
        trainer.fit(model, al_datamodule)

    # Saving results
    file_name = os.path.join(output_dir, 'results.json')
    with open(file_name, 'w', encoding='utf-8') as f:
       json.dump(results, f)


def build_model(num_epochs, num_classes, feature_size=None):
    model = deterministic.linear.LinearModel(feature_size, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, weight_decay=0.0, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = deterministic.DeterministicModel(
        model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy()},
    )

    return model


def build_al_strategy(name, subset_size,
                      kernel=None, alpha=None, precomputed=True, gamma=None, num_classes=None, train_features=None):
    if name == "random":
        query = random.RandomSampling()
    elif name == "entropy":
        query = uncertainty.EntropySampling(subset_size=subset_size)
    elif name == "coreset":
        query = coreset.CoreSet(subset_size=subset_size)
    elif name == "badge":
        query = badge.Badge(subset_size=subset_size)
    elif name == "typiclust":
        query = typiclust.TypiClust(subset_size=subset_size)
    elif name == "randomclust":
        query = randomclust.RandomClust(subset_size=subset_size)
    elif name == "xpal" or name == "xpalclust":
        if gamma == "calculate":
            gamma = _calculate_mean_gamma(train_features)
        else:
            gamma = gamma
        if precomputed:
            S = kernels(X=train_features, Y=train_features, metric=kernel, gamma=gamma)
        else:
            S = None

        if name == "xpal":
            query = xpal.XPAL(num_classes, S, subset_size=subset_size, alpha_c=alpha, alpha_x=alpha,
                              precomputed=precomputed, gamma=gamma, kernel=kernel)
        elif name == "xpalclust":
            query = xpalclust.XPALClust(num_classes, S, subset_size=subset_size, alpha_c=alpha, alpha_x=alpha,
                                        precomputed=precomputed, gamma=gamma, kernel=kernel)
    else:
        raise NotImplementedError(f"Active learning strategy {name} is not implemented!")
    return query


def build_dataset(name, path):
    if name == 'CIFAR10':
        data = datasets.CIFAR10(path)
    elif name == 'CIFAR100':
        data = datasets.CIFAR100(path)
    elif name == 'SVHN':
        data = datasets.SVHN(path)
    elif name == 'ImageNet100':
        data = datasets.ImageNet100(path)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented!")

    return data


strategies = ["random",
              "entropy",
              "badge",
              "typiclust",
              "xpal",
              "xpalclust"
              ]
seeds = [3, 4, 5, 6, 7, 8, 9]
acq_sizes = [10]
subset_sizes = [2500]

dataset_tuples = [("CIFAR10", "../../wide_resnet_28_10_CIFAR10_0.937.pth"),
                  ("CIFAR100", "../../wide_resnet_28_10_CIFAR100_0.682.pth")
                  ]
def main():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    tt = len(strategies) * len(seeds) * len(acq_sizes) * len(dataset_tuples) * len(subset_sizes)
    pbar = tqdm(product(dataset_tuples, subset_sizes, acq_sizes, strategies, seeds), total=tt, file=sys.stdout)
    for dataset, subset_size, acq_size, strategy, seed in pbar:
        pbar.set_description(
            f"Current experiment: strategy={strategy}, seed={seed}, subset_size={subset_size}, acq_size={acq_size}, dataset={dataset[0]}")
        output_dir = f"D:\\Dokumente\\Git\\dal-toolbox\\baselines\\active_learning\\notebooks\ynagel\\TimeTest\\{dataset[0]}\\{strategy}\\{acq_size}_{subset_size}\\seed{seed}"

        al_cycle_test(random_seed=seed,
                      strategy=strategy,
                      acq_size=acq_size,
                      data_file=dataset[1],
                      output_dir=output_dir,
                      subset_size=subset_size)


if __name__ == "__main__":
    main()
