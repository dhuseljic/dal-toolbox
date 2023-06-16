import os
import logging
import math
import json
import time
import hydra
import transformers
import torch
import wandb
from omegaconf import OmegaConf
# os.chdir(sys.path[0])

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import DataCollatorWithPadding

from dal_toolbox.active_learning.data import ALDataset
from dal_toolbox.active_learning.strategies import random, uncertainty, coreset, badge
from dal_toolbox.models import build_model
from dal_toolbox.utils import seed_everything
from dal_toolbox.datasets import build_al_datasets
from dal_toolbox.metrics.generalization import area_under_curve
from dal_toolbox.active_learning.strategies import cal
transformers.logging.set_verbosity_error()


@hydra.main(version_base=None, config_path="./configs", config_name="al_nlp")
def main(args):
    print(OmegaConf.to_yaml(args))
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    t_init = time.time()
    results = {}
    query_indices = {}
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        reinit=args.wandb.reinit,
        mode=args.wandb.mode,
        name=args.model.name+'_'+args.al_strategy.name+'_'+args.dataset.name+'#'+str(args.random_seed),
        config = OmegaConf.to_container(
            args, 
            resolve=True, 
            throw_on_missing=True
    ))

    writer =  SummaryWriter(log_dir=args.output_dir)

    # Setup Datasets
    logging.info('Building datasets. Creating random initial labeled pool with %s samples.',
            args.al_cycle.n_init)
    train_ds, query_ds, test_ds, ds_info = build_al_datasets(args)
    al_dataset = ALDataset(train_ds, query_ds, random_state=args.random_seed)

    if args.al_cycle.init_pool_file is not None: 
        logging.info('Using initial labeled pool from %s.', args.al_strategy.queried_indices_json)
        with open(args.al_cycle.init_pool_file, 'r', encoding='utf-8') as f:
            queried_indices_json = json.load(f)
            

    al_dataset.random_init(n_samples=args.al_cycle.n_init)

    # Setup Model
    logging.info('Building model: %s', args.model.name)
    model_dict = build_model(args, n_classes=ds_info['n_classes'])
    model = model_dict['model']
    train_one_epoch = model_dict['train']
    eval_one_epoch = model_dict["eval"]
    optimizer = model_dict['train_kwargs']['optimizer']
    initial_states = model_dict['initial_states']
    data_collator = DataCollatorWithPadding(
        tokenizer=ds_info['tokenizer'],
        padding = 'longest',
        return_tensors="pt",
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.model.batch_size*4,
        shuffle=False,
        collate_fn=data_collator
    )

    # Setup Query
    logging.info('Building query strategy %s', args.al_strategy.name)
    al_strategy = build_query(args, device=args.device)

    for i_acq in range(0, args.al_cycle.n_acq + 1):
        logging.info('Starting AL iteration %s / %s', i_acq, args.al_cycle.n_acq)
        t_start = time.time()
        print(f'Starting Al iteration {i_acq}')
        print(f'Train Dataset: {len(al_dataset.labeled_indices)}')
        print(f'Pool Dataset available: {len(al_dataset.unlabeled_dataset)}')

        cycle_results = {}
        if i_acq != 0:
            logging.info('Querying %s samples with strategy `%s`', args.al_cycle.acq_size, args.al_strategy.name)
            print('> Querying.')
            indices = al_strategy.query(
                model=model,
                dataset=al_dataset.query_dataset,
                unlabeled_indices = al_dataset.unlabeled_indices,
                labeled_indices = al_dataset.labeled_indices,
                acq_size=args.al_cycle.acq_size,
                collator=data_collator
            )
            al_dataset.update_annotations(indices)
            query_time = time.time() - t_start
            logging.info('Querying time was %.2f minutes', query_time/60)
            cycle_results['query_indices'] = indices
            cycle_results['query_time'] = query_time

        logging.info('Training on labeled pool with %s samples', len(al_dataset.labeled_dataset))
        print('> Training.')
        t_start = time.time()
        train_history = []
        train_loader = DataLoader(
            al_dataset.labeled_dataset,
            batch_size=args.model.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=True      
        )
        optimizer.load_state_dict(initial_states['optimizer'])

        #TODO: find a way to outsource the trainloader (does not work in model because of num_train_steps)
        if args.model.optimizer.lr_scheduler == "linear_warmup":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=math.ceil(args.model.n_epochs * len(train_loader) * args.model.optimizer.warmup_ratio),
                num_training_steps=args.model.n_epochs * len(train_loader)
            )

        elif args.model.optimizer.lr_scheduler == "constant_warmup":        
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
            )

        if args.al_cycle.cold_start:
            model.load_state_dict(initial_states['model'])
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
        training_time = time.time() - t_start
        logging.info('Training took %.2f minutes', training_time/60)
        logging.info('Training stats: %s', train_stats)
        cycle_results['train_history'] = train_history
        cycle_results['training_time'] = training_time

        print('> Evaluation.')
        logging.info('Evaluation with %s sample', len(test_ds))
        t_start = time.time()
        test_stats = eval_one_epoch(
            model=model,
            dataloader=test_loader,
            epoch=i_epoch,
            **model_dict['eval_kwargs']
        )

        evaluation_time = time.time()- t_start
        logging.info('Evaluation took %.2f minutes', evaluation_time/60)
        logging.info('Evaluation stats: %s', test_stats)
        cycle_results['evaluation_time'] = evaluation_time
        cycle_results['test_stats'] = test_stats

        for key, value in test_stats.items():
            wandb.log(
                {key: value},
                step=len(al_dataset.labeled_indices)
            )
            writer.add_scalar(
                tag=f"test_stats/{key}",
                scalar_value=value,
                global_step=len(al_dataset.labeled_indices))
             
        cycle_results.update({
            "labeled_indices": al_dataset.labeled_indices,
            "n_labeled_samples": len(al_dataset.labeled_dataset),
            "unlabeled_indices": al_dataset.unlabeled_indices,
            "n_unlabeled_samples": len(al_dataset.unlabeled_dataset),
        })

        results[f'cycle{i_acq}'] = cycle_results

        # checkpoint
        logging.info('Saving checkpoint for cycle %s', i_acq)
        if i_acq % args.check_freq_model == 0:
            checkpoint = {
                "args": args,
                "model": model.state_dict(),
                "al_dataset": al_dataset.state_dict(),
                "optimizer": model_dict['train_kwargs']['optimizer'].state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                "cycle_results": cycle_results,
            }
            torch.save(checkpoint, os.path.join(os.getcwd(), f"check{i_acq}.pth"))   
        
    writer.close()
    acc = [cycles['test_stats']['test_batch_acc_epoch'] for cycles in results.values()]
    f1 = [cycles['test_stats']['test_batch_f1_epoch'] for cycles in results.values()]
    acc_blc = [cycles['test_stats']['test_batch_acc_balanced_epoch'] for cycles in results.values()]
    auc_acc = area_under_curve(acc)
    auc_f1 = area_under_curve(f1)
    auc_acc_blc = area_under_curve(acc_blc)
    results[f'cycle{i_acq}']['test_stats']['auc_acc'] = auc_acc
    results[f'cycle{i_acq}']['test_stats']['auc_f1'] = auc_f1
    results[f'cycle{i_acq}']['test_stats']['auc_acc_blc'] = auc_acc_blc

    wandb.log({
        'final_auc_acc':auc_acc,
        'final_auc_f1': auc_f1,
        'final_auc_acc_blc': auc_acc_blc
        })
    
    # save all results
    savepath = os.path.join(args.output_dir, 'results.json')
    logging.info('Saving results to %s', savepath)
    print(f'Saving results to {savepath}.')
    with open(savepath, 'w', encoding='utf8') as file:
        json.dump(results, file)
    
    # save only indices
    savepath = os.path.join(args.output_dir, 'queried_indices.json')
    logging.info('Saving results to %s', savepath)
    print(f'Saving queried indices to {savepath}.')
    with open(savepath, 'w', encoding='utf8') as file:
        json.dump(queried_indices, file)

    time_overall = time.time()- t_init
    logging.info('Experiment took %.2f minutes', time_overall/60)

def build_query(args, **kwargs):
    if args.al_strategy.name == "random":
        query = random.RandomSampling(random_seed=args.random_seed)
    elif args.al_strategy.name == "uncertainty":
        device = kwargs['device']
        query = uncertainty.UncertaintySampling(
            uncertainty_type=args.al_strategy.uncertainty_type,
            subset_size=args.dataset.train_subset,
            device=device,
        )
    elif args.al_strategy.name == "coreset":
        device = kwargs['device']
        query = coreset.CoreSet(subset_size=args.dataset.train_subset, device=device)
    elif args.al_strategy.name == "badge":
        device = kwargs['device']
        query = badge.Badge(subset_size=args.dataset.train_subset, device=device)
    elif args.al_strategy.name == "cal":
        device = kwargs['device']
        query = cal.CAL(subset_size=args.dataset.train_subset, device=device)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query

if __name__ == "__main__":
    main()
