"""This scripts creates the json files which store the initial pools use throughout the evaluation."""
import os
import json
import random

from dal_toolbox import datasets

num_reps = 11
size = 128
dataset_name = 'CIFAR10'
dataset_path = '/datasets'
output_dir = os.path.dirname(__file__)

if dataset_name == 'CIFAR10':
    data = datasets.CIFAR10(dataset_path)
elif dataset_name == 'CIFAR100':
    data = datasets.CIFAR100(dataset_path)
elif dataset_name == 'SVHN':
    data = datasets.SVHN(dataset_path)
else:
    raise NotImplementedError

for i_rep in range(num_reps):
    rng = random.Random(i_rep)
    unlabeled_indices = range(len(data.train_dataset))
    indices = rng.sample(unlabeled_indices, k=size)

    path = os.path.join(output_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    file_name =  os.path.join(path, f'random_{size}_seed{i_rep}.json')
    print("Saving initial pool indices to", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(indices, f, sort_keys=False)
