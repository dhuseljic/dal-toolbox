import copy
import torch
import torch.nn as nn

from .deterministic import bert, distilbert, roberta, lenet, resnet, wide_resnet, wide_resnet_new
from .deterministic import train as train_deterministic
from .deterministic import evaluate as eval_deterministic

from .mc_dropout import resnet as resnet_mcdropout
from .mc_dropout import wide_resnet as wide_resnet_mcdropout
from .mc_dropout import train as train_mcdropout
from .mc_dropout import evaluate as eval_mcdropout

from .ensemble import voting_ensemble
from .ensemble import train as train_ensemble
from .ensemble import evaluate as eval_ensemble

from .sngp import resnet as resnet_sngp
from .sngp import wide_resnet as wide_resnet_sngp
from .sngp import train as train_sngp
from .sngp import evaluate as eval_sngp


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']
    if args.model.name == 'resnet18_labelsmoothing':
        model = resnet.ResNet18(n_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.n_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.model.label_smoothing)
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': train_deterministic.train_one_epoch,
            'evaluate': eval_deterministic.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'wideresnet2810_deterministic':
        model_dict = build_wide_resnet_deterministic(
            n_classes=n_classes,
            dropout_rate=args.model.dropout_rate,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            n_epochs=args.model.n_epochs,
            device=args.device,
        )

    elif args.model.name == 'wideresnet2810_mcdropout':
        model_dict = build_wide_resnet_mcdropout(
            n_classes=n_classes,
            n_mc_passes=args.model.n_passes,
            dropout_rate=args.model.dropout_rate,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            n_epochs=args.model.n_epochs,
            device=args.device,
        )

    elif args.model.name == 'wideresnet2810_sngp':
        model_dict = build_wide_resnet_sngp(
            n_classes=n_classes,
            input_shape=(3, 32, 32),
            depth=28,
            widen_factor=10,
            use_spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.norm_bound,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            scale_random_features=args.model.gp.scale_random_features,
            random_feature_type=args.model.gp.random_feature_type,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
            dropout_rate=args.model.dropout_rate,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            n_epochs=args.model.n_epochs,
            device=args.device,
        )

    elif args.model.name == 'wideresnet2810_ensemble':
        model_dict = build_wide_resnet_ensemble(
            n_classes=n_classes,
            n_member=args.model.n_member,
            dropout_rate=args.model.dropout_rate,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            n_epochs=args.model.n_epochs,
            device=args.device
        )

    elif args.model.name == 'lenet_deterministic':
        model_dict = build_lenet_deterministic(
            n_classes=n_classes,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            n_epochs=args.model.n_epochs,
            device=args.device
        )

    elif args.model.name == 'bert':
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
        NotImplementedError(f'Model {args.model} not implemented.')

    return model_dict


def build_lenet_deterministic(n_classes, lr, weight_decay, momentum, n_epochs, device):
    model = lenet.LeNet(num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    model_dict = {
        'model': model,
        'optimizer': optimizer,
        'train_one_epoch': train_deterministic.train_one_epoch,
        'evaluate': eval_deterministic.evaluate,
        'lr_scheduler': lr_scheduler,
        'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=device),
        'eval_kwargs': dict(criterion=criterion, device=device),
    }
    return model_dict


def build_wide_resnet_deterministic(n_classes, dropout_rate, lr, weight_decay, momentum, n_epochs, device):
    model = wide_resnet.wide_resnet_28_10(num_classes=n_classes, dropout_rate=dropout_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    model_dict = {
        'model': model,
        'optimizer': optimizer,
        'train_one_epoch': train_deterministic.train_one_epoch,
        'evaluate': eval_deterministic.evaluate,
        'lr_scheduler': lr_scheduler,
        'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=device),
        'eval_kwargs': dict(criterion=criterion, device=device),
    }
    return model_dict


def build_wide_resnet_mcdropout(n_classes, n_mc_passes, dropout_rate, lr, weight_decay, momentum, n_epochs, device):
    model = wide_resnet_mcdropout.dropout_wide_resnet_28_10(n_classes, n_mc_passes, dropout_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    model_dict = {
        'model': model,
        'optimizer': optimizer,
        'train_one_epoch': wide_resnet_mcdropout.train_one_epoch,
        'evaluate': wide_resnet_mcdropout.evaluate,
        'lr_scheduler': lr_scheduler,
        'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=device),
        'eval_kwargs': dict(criterion=criterion, device=device),
    }
    return model_dict


def build_wide_resnet_ensemble(n_classes, n_member, dropout_rate, lr, weight_decay, momentum, n_epochs, device):
    members, lr_schedulers, optimizers = [], [], []
    for _ in range(n_member):
        member = wide_resnet.wide_resnet_28_10(num_classes=n_classes, dropout_rate=dropout_rate)
        optimizer = torch.optim.SGD(member.parameters(), lr=lr, weight_decay=weight_decay,
                                    momentum=momentum, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        members.append(member)
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    model = voting_ensemble.Ensemble(members)
    optimizer = voting_ensemble.EnsembleOptimizer(optimizers)
    lr_scheduler = voting_ensemble.EnsembleLRScheduler(lr_schedulers)
    criterion = nn.CrossEntropyLoss()
    model_dict = {
        'model': model,
        'optimizer': optimizer,
        'train_one_epoch': train_ensemble.train_one_epoch,
        'evaluate': eval_ensemble.evaluate,
        'lr_scheduler': lr_scheduler,
        'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=device),
        'eval_kwargs': dict(criterion=criterion, device=device),
    }
    return model_dict


def build_wide_resnet_sngp(n_classes, input_shape, depth, widen_factor, dropout_rate, use_spectral_norm, norm_bound,
                           n_power_iterations, num_inducing, kernel_scale, scale_random_features, random_feature_type,
                           mean_field_factor, cov_momentum, ridge_penalty, lr, weight_decay, momentum, n_epochs, device):
    model = wide_resnet_sngp.WideResNetSNGP(
        depth=depth,
        widen_factor=widen_factor,
        dropout_rate=dropout_rate,
        num_classes=n_classes,
        input_shape=input_shape,
        spectral_norm=use_spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_inducing=num_inducing,
        kernel_scale=kernel_scale,
        normalize_input=False,
        scale_random_features=scale_random_features,
        random_feature_type=random_feature_type,
        mean_field_factor=mean_field_factor,
        cov_momentum=cov_momentum,
        ridge_penalty=ridge_penalty,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    model_dict = {
        'model': model,
        'optimizer': optimizer,
        'train_one_epoch': wide_resnet_sngp.train_one_epoch,
        'evaluate': wide_resnet_sngp.evaluate,
        'lr_scheduler': lr_scheduler,
        'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=device),
        'eval_kwargs': dict(criterion=criterion, device=device),
    }
    return model_dict