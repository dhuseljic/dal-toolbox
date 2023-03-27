import ray
import ray.tune as tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.repeater import Repeater


import torch
import torch.nn as nn

from dal_toolbox.datasets import cifar
from dal_toolbox.models import deterministic


def objective(config):
    device = 'cuda'
    seed = config['__trial_index__']

    torch.manual_seed(seed)
    ds_path = '/datasets'
    train_ds, ds_info = cifar.build_cifar10('train', ds_path, return_info=True)
    val_ds = cifar.build_cifar10('test', ds_path)

    model = nn.Sequential(nn.Flatten(), nn.Linear((32*32*3), ds_info['n_classes']))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=.01, momentum=.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    trainer = deterministic.trainer.DeterministicTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=None
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, sampler=torch.randperm(len(train_ds))[:100])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, sampler=torch.randperm(len(val_ds))[:1000])

    trainer.train(100, train_loader=train_loader)
    test_stats = trainer.evaluate(dataloader=val_loader)

    return test_stats


def main():
    distributed = False
    n_reps = 3
    n_samples = 10

    objective_with_resources = tune.with_resources(objective, resources={'cpu': 8, 'gpu': 1})
    if distributed:
        ray.init(address='auto')
    search_alg = BayesOptSearch()
    search_alg = Repeater(search_alg, repeat=n_reps)
    tune_config = tune.TuneConfig(search_alg=search_alg, num_samples=n_samples*n_reps, metric="test_acc1", mode="max")
    search_space = {
        "lr": tune.uniform(0, 1),
        # "lr": tune.grid_search([0.001, 0.01, 0.1]),
        # "weight_decay": tune.grid_search([0.0005, 0.005, .05]),
    }
    tuner = tune.Tuner(objective_with_resources, param_space=search_space, tune_config=tune_config)
    results = tuner.fit()
    print(results.get_best_result(metric="test_acc1", mode="max").config)


if __name__ == '__main__':
    main()
