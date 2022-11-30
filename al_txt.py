#%%
import os 
import copy
import math
import json
import hydra
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from active_learning.data import ALDataset
from active_learning.strategies import random, uncertainty

from models import build_model
from utils import seed_everything
from data import build_al_datasets
from active_learning import build_query

@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    print(OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    writer = SummaryWriter(log_dir=args.output_dir)

    train_ds, query_ds, test_ds, n_classes = build_al_datasets(args)
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
        batch_size=args.val_batch_size, 
        shuffle=True
    )

    for i_acq in range(0, args.al_cycle.n_acq + 1):
        print(f'Starting Al iteration {i_acq}')

        if i_acq != 0:
            print('> Querying.')

            indices = al_strategy.query(
                model=model, 
                dataset=al_dataset, 
                acq_size=args.al_cycle.acq_size,
                batch_size=args.val_batch_size,
                device=args.device
            )

            al_dataset.update_annotations(indices)

        # set initial states
        optimizer.load_state_dict(initial_states['optimizer'])
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(len(al_dataset.labeled_dataset) * args.n_epochs * 0.1),
            num_trainings_steps = len(al_dataset.labeled_dataset) * args.n_epochs
        )
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_states['model'])

        print('> Training.')
        train_history = []
        train_loader = DataLoader(
            al_dataset.labeled_dataset, 
            batch_size=args.dataset.batch_size,
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
            'n_labeled_samples': int(len(al_dataset.labeled_dataset)),
            'unlabeled_indices': al_dataset.unlabeled_indices, 
            'n_unlabeled_indices': int(len(al_dataset.unlabeled_indices))
        })
        writer.close()
        savepath = os.path.join(args.output_dir, 'results.json')
        print(f'Savubg results to {savepath}.')
        with open(savepath, 'w') as f:
            json.dump(history, f)

if __name__ == "__main__":
    main()


