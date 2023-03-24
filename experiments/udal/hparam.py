import ray
from ray import tune

import torch
import torch.nn as nn

from dal_toolbox.datasets import cifar
from dal_toolbox.models import deterministic


def objective(config):
    torch.manual_seed(0)
    train_ds, ds_info = cifar.build_cifar10('train', '/tmp', return_info=True)
    val_ds = cifar.build_cifar10('test', '/tmp')

    model = deterministic.resnet.ResNet18(ds_info['n_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=.9, nesterov=True,)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    trainer = deterministic.trainer.DeterministicTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device='cpu',
        output_dir=None
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, sampler=torch.randperm(len(train_ds))[:2000])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, sampler=torch.randperm(len(val_ds))[:1000])

    trainer.train(100, train_loader=train_loader)
    test_stats = trainer.evaluate(dataloader=val_loader)

    return test_stats


# 2. Define a search space.
search_space = {
    "lr": tune.grid_search([0.01, 0.1]),
    "weight_decay": tune.grid_search([0.005, .05]),
}

# Parallelism is automatic, split accross devices. Assume I have 16 cpus, specifying 4 cpu in resources will allow 4 parallel tunings
objective_with_resources = tune.with_resources(objective, resources={'cpu': 4, 'gpu': 0})
# ray.init(address='141.51.131.177:8080')
ray.init(address='141.51.125.49:9876')
tuner = tune.Tuner(
    objective_with_resources,
    param_space=search_space,
)
results = tuner.fit()
print(results.get_best_result(metric="test_acc1", mode="max").config)
