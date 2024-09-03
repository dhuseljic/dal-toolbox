"""This scripts creates the json files which store the initial pools use throughout the evaluation."""
import os
import json
import random
from datasets import load_dataset

def load_ds(name):
    print('>> Loading dataset')
    try:
        if name == 'wikitalk':
            ds = load_dataset('jigsaw_toxicity_pred', data_dir= '~/.cache/huggingface/manual_ds/jigsaw_toxicity_pred')
        else:
            ds = load_dataset(name)

    except FileNotFoundError:
        ds = load_dataset('glue', name)
    
    return ds

n_reps = 5
size = 100
dataset_names = ['agnews', 'banks77', 'dbpedia', 'fnc1', 'imdb', 'mnli', 'qnli', 'sst2', 'trec6', 'wikitalk', 'yelp5']
output_dir = '.'

for dataset_name in dataset_names:
    if dataset_name == 'agnews':
        complete_ds = load_ds('ag_news')
        query_ds = complete_ds['train']

    elif dataset_name == 'banks77':
        complete_ds = load_ds('banking77')
        query_ds = complete_ds['train']

    elif dataset_name == 'dbpedia':
        complete_ds = load_ds('dbpedia_14')
        query_ds = complete_ds['train']

    elif dataset_name == 'fnc1':
        complete_ds = load_ds('nid989/FNC-1')
        query_ds = complete_ds['train']

    elif dataset_name == 'mnli':
        complete_ds = load_ds('multi_nli')
        query_ds = complete_ds['train']

    elif dataset_name == 'qnli':
        complete_ds = load_ds('qnli')
        query_ds = complete_ds['train']

    elif dataset_name == 'sst2':
        complete_ds = load_ds('sst2')
        query_ds = complete_ds['train']

    elif dataset_name == 'trec6':
        complete_ds = load_ds('trec')
        query_ds = complete_ds['train']

    elif dataset_name == 'wikitalk':
        complete_ds =load_dataset('jigsaw_toxicity_pred', data_dir= '~/.cache/huggingface/manual_ds/jigsaw_toxicity_pred')
        query_ds = complete_ds['train']

    elif dataset_name == 'yelp5':
        complete_ds = load_ds('yelp_review_full')
        query_ds = complete_ds['train']

    else:
        NotImplementedError()

    for i_rep in range(0, n_reps+1):
        rng = random.Random(i_rep)
        unlabeled_indices = range(len(query_ds))
        indices = rng.sample(unlabeled_indices, k=size)
        indices.sort()

        path = os.path.join(output_dir, dataset_name)
        os.makedirs(path, exist_ok=True)
        file_name =  os.path.join(path, f'random_{size}_seed{i_rep}.json')
        print("Saving initial pool indices to", file_name)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(indices, f, sort_keys=False)


