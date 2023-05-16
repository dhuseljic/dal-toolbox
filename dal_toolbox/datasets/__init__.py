import torch
from torch.utils.data import Subset
from .activeglae import agnews, banks77, dbpedia, fnc1, mnli, qnli, sst2, trec6, wikitalk, yelp5


def build_al_datasets(args):
    if args.dataset.name == 'imdb':
        complete_ds, ds_info = imdb.build_imdb(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'agnews':
        complete_ds, ds_info = agnews.build_agnews(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'banks77':
        complete_ds, ds_info = banks77.build_banks77(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'dbpedia':
        complete_ds, ds_info = dbpedia.build_dbpedia(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'fnc1':
        complete_ds, ds_info = fnc1.build_fnc1(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'mnli':
        complete_ds, ds_info = mnli.build_mnli(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "validation_matched")

    elif args.dataset.name == 'qnli':
        complete_ds, ds_info = qnli.build_qnli(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "validation")

    elif args.dataset.name == 'sst2':
        complete_ds, ds_info = sst2.build_sst2(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "validation")

    elif args.dataset.name == 'trec6':
        complete_ds, ds_info = trec6.build_trec6(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'wikitalk':
        complete_ds, ds_info = wikitalk.build_wikitalk(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    elif args.dataset.name == 'yelp5':
        complete_ds, ds_info = yelp5.build_yelp5(args)
        train_ds = complete_ds['train']
        query_ds = complete_ds['train']
        test_ds_id = create_testsubset(complete_ds, args, "test")

    else:
        raise NotImplementedError('Dataset not available')

    return train_ds, query_ds, test_ds_id, ds_info


def equal_set_sizes(ds_id, ds_ood):
    # Make test id and ood the same size
    n_samples_id = len(ds_id)
    n_samples_ood = len(ds_ood)
    if n_samples_id < n_samples_ood:
        rnd_indices = torch.randperm(n_samples_ood)[:n_samples_id]
        ds_ood = Subset(ds_ood, indices=rnd_indices)
    elif n_samples_id > n_samples_ood:
        rnd_indices = torch.randperm(n_samples_id)[:n_samples_ood]
        ds_id = Subset(ds_id, indices=rnd_indices)
    return ds_id, ds_ood


def create_testsubset(complete_ds, args, hf_name):
    if args.dataset.test_subset:
        test_ds_id = complete_ds[hf_name].shuffle(seed=args.random_seed.select).select(range(args.dataset.test_subset))
    else:
        test_ds_id = complete_ds[hf_name]
    return test_ds_id
