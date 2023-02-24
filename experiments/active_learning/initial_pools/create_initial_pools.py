"""This scripts creates the json files which store the initial pools use throughout the evaluation."""
import os
import json
import random

from dal_toolbox import datasets


n_reps = 10
size = 100
dataset_name = 'CIFAR100'
output_dir = '.'


if dataset_name == 'CIFAR10':
    ds = datasets.cifar.build_cifar10('query', '/tmp')
elif dataset_name == 'CIFAR100':
    ds = datasets.cifar.build_cifar100('query', '/tmp')
elif dataset_name == 'SVHN':
    ds = datasets.svhn.build_svhn('query', '/tmp')
else:
    NotImplementedError()

for i_rep in range(n_reps):
    rng = random.Random(i_rep)
    unlabeled_indices = range(len(ds))
    indices = rng.sample(unlabeled_indices, k=size)

    path = os.path.join(output_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    file_name =  os.path.join(path, f'random_{size}_seed{i_rep}.json')
    print("Saving initial pool indices to", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(indices, f, sort_keys=False)
