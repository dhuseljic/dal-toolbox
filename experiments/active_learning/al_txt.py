#%%
import os 
import copy
import math
import json
import hydra
from omegaconf import OmegaConf
import sys
sys.path.append('.') #?

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()
from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.active_learning.strategies import random, uncertainty

from dal_toolbox.models import build_model
from dal_toolbox.utils import seed_everything
from dal_toolbox.datasets import build_al_datasets

@hydra.main(version_base=None, config_path="./configs", config_name="active_learning_nlp")
def main(args):
    print(OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    writer = SummaryWriter(log_dir=args.output_dir)

    train_ds, query_ds, test_ds, ds_info = build_al_datasets(args)
    test_ds = test_ds.shuffle(seed=42).select(range(500))
    al_dataset = ALDataset(train_ds, query_ds)
    al_dataset.random_init(n_samples=args.al_cycle.n_init)

    model_dict = build_model(
        args,
        n_classes=n_classes,
        train_ds = al_dataset.train_dataset
    )
    model = model_dict['model']
    train_one_epoch = model_dict['train']
    eval_one_epoch = model_dict["eval"]
    optimizer = model_dict['train_kwargs']['optimizer']
    initial_states = model_dict['initial_states']

    al_strategy = build_query(args)
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.model.batch_size*4, 
        shuffle=False
    )

    for i_acq in range(0, args.al_cycle.n_acq + 1):
        print(f'Starting Al iteration {i_acq}')
        print(f'Train Dataset: {len(al_dataset.labeled_indices)}')
        print(f'Pool Dataset available: {len(al_dataset.unlabeled_dataset)}')

        if i_acq != 0:
            print('> Querying.')

            indices = al_strategy.query(
                model=model, 
                dataset=al_dataset, 
                acq_size=args.al_cycle.acq_size,
                batch_size=args.model.batch_size*4,
                device=args.device
            )

            al_dataset.update_annotations(indices)

        # set initial states
        optimizer.load_state_dict(initial_states['optimizer'])
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(len(al_dataset.labeled_dataset) * args.model.n_epochs * 0.1),
            num_training_steps = len(al_dataset.labeled_dataset) * args.model.n_epochs
        )
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_states['model'])

        print('> Training.')
        train_history = []

        train_loader = DataLoader(
            al_dataset.labeled_dataset, 
            batch_size=args.model.batch_size,
            shuffle=True          
        )
        #TODO: wird hier nicht der falsche optimizer übergeben?
        for i_epoch in range(args.model.n_epochs):
            train_stats = train_one_epoch(
                model=model,
                dataloader=train_loader,
                epoch=i_epoch,
                scheduler=lr_scheduler,
                **model_dict['train_kwargs']
            )

            for key, value in train_stats.items():
                writer.add_scalar(
                    tag=f"cycle_{i_acq}_train/{key}", 
                    scalar_value=value, 
                    global_step=i_epoch
                    )
            train_history.append(train_stats)
        
        print('> Evaluation.')
        test_stats = eval_one_epoch(
            model=model,
            dataloader=test_loader,
            epoch = i_epoch,
            **model_dict['eval_kwargs']
        )

        for key, value in test_stats.items():
            writer.add_scalar(
                tag=f"test_stats/{key}", 
                scalar_value=value, 
                global_step=i_acq)
        
        history.append({
            'train_history': train_history,
            'test_stats': test_stats,
            'labeled_indices': al_dataset.labeled_indices,
            'n_labeled_samples': len(al_dataset.labeled_dataset),
            'unlabeled_indices': al_dataset.unlabeled_indices, 
            'n_unlabeled_indices': len(al_dataset.unlabeled_indices)
        })
        writer.close()
        savepath = os.path.join(args.output_dir, 'results.json')
        print(f'Saving results to {savepath}.')
        with open(savepath, 'w') as f:
            json.dump(history, f)

def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    elif args.al_strategy.name == "uncertainty":
        device = kwargs['device']
        query = uncertainty.UncertaintySampling(
            uncertainty_type=args.al_strategy.uncertainty_type,
            subset_size=args.al_strategy.subset_size,
            device=device,
        )
    elif args.al_strategy.name == "coreset":
        device = kwargs['device']
        query = coreset.CoreSet(subset_size=args.al_strategy.subset_size, device=device)
    elif args.al_strategy.name == "badge":
        device = kwargs['device']
        query = badge.Badge(subset_size=args.al_strategy.subset_size, device=device)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query

if __name__ == "__main__":
    main()


