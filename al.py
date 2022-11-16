import os
import copy
import json
import hydra

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from active_learning.data import ALDataset
from models import build_model, build_eval_model
from utils import seed_everything
from datasets import build_al_datasets


from active_learning.strategies import random, uncertainty, bayesian_uncertainty


@hydra.main(version_base=None, config_path="./configs", config_name="active_learning")
def main(args):
    # Display Args
    print(args)

    # Enabeling reproducability
    seed_everything(args.random_seed)

    # Create Logging dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Necessary for logging
    history = []
    writer = SummaryWriter(log_dir=args.output_dir)

    # Setup Dataset
    train_ds, query_ds, val_ds, n_classes = build_al_datasets(args)
    al_dataset = ALDataset(train_ds, query_ds)
    al_dataset.random_init(n_samples=args.al_cycle.n_init)

    # Setup Model
    #TODO: Model could be renamed into query_model for a more understandable code?
    model_dict = build_model(args, n_classes=n_classes, train_ds=al_dataset.labeled_dataset)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, lr_scheduler = model_dict['optimizer'], model_dict['lr_scheduler']

    # Setup Eval Model
    eval_model_dict = build_eval_model(args, n_classes=n_classes)
    eval_model, eval_train_one_epoch, eval_evaluate = eval_model_dict['model'], eval_model_dict['train_one_epoch'], eval_model_dict['evaluate']
    eval_optimizer, eval_lr_scheduler = eval_model_dict['optimizer'], eval_model_dict['lr_scheduler']

    # Setup Query
    al_strategy = build_query(args)

    # Setup initial states
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
    initial_eval_model_state = copy.deepcopy(eval_model.state_dict())
    initial_eval_optimizer_state = copy.deepcopy(eval_optimizer.state_dict())
    initial_eval_scheduler_state = copy.deepcopy(eval_lr_scheduler.state_dict())

    # Active Learning Cycles
    for i_acq in range(0, args.al_cycle.n_acq + 1):
        print(f'Starting AL iteration {i_acq}')

        # Analyse unlabeled set and query most promising data
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

        # Setup Training: If cold start is set, reset the model parameters
        optimizer.load_state_dict(initial_optimizer_state)
        eval_optimizer.load_state_dict(initial_eval_optimizer_state)
        lr_scheduler.load_state_dict(initial_scheduler_state)
        eval_lr_scheduler.load_state_dict(initial_eval_scheduler_state)
        if args.al_cycle.cold_start:
            model.load_state_dict(initial_model_state)
            eval_model.load_state_dict(initial_eval_model_state)

        # Train with updated Annotations
        print('> Training. (model)')
        train_history = []
        for i_epoch in range(args.model.n_epochs):
            drop_last = args.model.batch_size < len(al_dataset.labeled_dataset)
            train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.model.batch_size, shuffle=True, drop_last=drop_last)
            train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
            if lr_scheduler:
                lr_scheduler.step()

            for key, value in train_stats.items():
                writer.add_scalar(tag=f"cycle_{i_acq}_train/{key}", scalar_value=value, global_step=i_epoch)
            train_history.append(train_stats)

        # Train the eval model seperatly
        print('> Training. (eval_model)')
        eval_train_history = []
        for i_epoch in range(args.eval_model.n_epochs):
            drop_last = args.eval_model.batch_size < len(al_dataset.labeled_dataset)
            train_loader = DataLoader(al_dataset.labeled_dataset, batch_size=args.eval_model.batch_size, shuffle=True, drop_last=drop_last)
            train_stats = train_one_epoch(eval_model, train_loader, **eval_model_dict['train_kwargs'], epoch=i_epoch)
            if eval_lr_scheduler:
                eval_lr_scheduler.step()

            for key, value in train_stats.items():
                writer.add_scalar(tag=f"cycle_{i_acq}_eval_model_train/{key}", scalar_value=value, global_step=i_epoch)
            eval_train_history.append(train_stats)

        # Evaluate train_model and eval_model
        #TODO: Is it still necessary to evaluate the query model on test_data?
        print('> Evaluation.')
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=True)
        test_stats = evaluate(model, val_loader, dataloaders_ood={}, **model_dict['eval_kwargs'])
        eval_test_stats = evaluate(eval_model, val_loader, dataloaders_ood={}, **eval_model_dict['eval_kwargs'])
        print(test_stats)

        # Log
        for key, value in test_stats.items():
            writer.add_scalar(tag=f"test_stats/{key}", scalar_value=value, global_step=i_acq)
        history.append({
            "train_history": train_history,
            "eval_train_history": eval_train_history,
            "test_stats": test_stats,
            "eval_test_stats": eval_test_stats,
            "labeled_indices": al_dataset.unlabeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })

    # Save results
    fname = os.path.join(args.output_dir, 'results.json')
    print(f"Saving results to {fname}.")
    # torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
    with open(fname, 'w') as f:
        json.dump(history, f)


def build_query(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "uncertainty":
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query


if __name__ == "__main__":
    main()
