import copy
import torch
import torch.nn as nn

from .deterministic import bert, distilbert, roberta
from .deterministic import train as train_deterministic
from .deterministic import evaluate as eval_deterministic


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']

    if args.model.name == 'bert':
        model = bert.BertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=n_classes
        )

        if args.model.optimizer.name == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
            )

        elif args.model.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay
            )

        else:
            raise NotImplementedError(f'{args.model.optimizer.name} not implemented')

        criterion = nn.CrossEntropyLoss()
        train_kwargs = {
            'optimizer': optimizer,
            'criterion': criterion,
            'device': args.device
        }
        eval_kwargs = {
            'criterion': criterion,
            'device': args.device
        }
        initial_states = {
            'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict())
        }
        # TODO: LR SCHEDULER?

        model_dict = {
            'model': model,
            'train': train_deterministic.train_one_epoch_bertmodel,
            'eval': eval_deterministic.evaluate_bertmodel,
            'train_kwargs': train_kwargs,
            'eval_kwargs': eval_kwargs,
            'initial_states': initial_states
        }

    elif args.model.name == 'roberta':
        model = roberta.RoBertaSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=n_classes
        )

        if args.model.optimizer.name == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
            )

        elif args.model.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay
            )

        else:
            raise NotImplementedError(f'{args.model.optimizer.name} not implemented')

        criterion = nn.CrossEntropyLoss()
        train_kwargs = {
            'optimizer': optimizer,
            'criterion': criterion,
            'device': args.device
        }
        eval_kwargs = {
            'criterion': criterion,
            'device': args.device
        }
        initial_states = {
            'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict())
        }
        # TODO: LR SCHEDULER?

        model_dict = {
            'model': model,
            'train': train_deterministic.train_one_epoch_bertmodel,
            'eval': eval_deterministic.evaluate_bertmodel,
            'train_kwargs': train_kwargs,
            'eval_kwargs': eval_kwargs,
            'initial_states': initial_states
        }

    elif args.model.name == 'distilbert':
        model = distilbert.DistilbertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=n_classes
        )

        if args.model.optimizer.name == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
            )

        elif args.model.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay
            )

        else:
            raise NotImplementedError(f'{args.model.optimizer.name} not implemented')

        criterion = nn.CrossEntropyLoss()
        train_kwargs = {
            'optimizer': optimizer,
            'criterion': criterion,
            'device': args.device
        }
        eval_kwargs = {
            'criterion': criterion,
            'device': args.device
        }
        initial_states = {
            'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict())
        }
        # TODO: LR SCHEDULER?

        model_dict = {
            'model': model,
            'train': train_deterministic.train_one_epoch_bertmodel,
            'eval': eval_deterministic.evaluate_bertmodel,
            'train_kwargs': train_kwargs,
            'eval_kwargs': eval_kwargs,
            'initial_states': initial_states
        }

    else:
        raise NotImplementedError(f'Model {args.model.name} not implemented.')

    return model_dict
