import torch
import torch.nn as nn

from . import resnet, resnet_ensemble, resnet_mcdropout, resnet_sngp 
from . import wideresnet2810, wideresnet2810_sngp


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']
    if args.model.name == 'resnet18_deterministic':
        model = resnet.ResNet18(n_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet.train_one_epoch,
            'evaluate': resnet.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'resnet18_mcdropout':
        model = resnet_mcdropout.DropoutResNet18(n_classes, args.model.n_passes, args.model.dropout_rate)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet_mcdropout.train_one_epoch,
            'evaluate': resnet_mcdropout.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'resnet18_ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = resnet_ensemble.ResNet18(n_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
                nesterov=True
            )
            lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = resnet_ensemble.Ensemble(members)
        optimizer = resnet_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = resnet_ensemble.EnsembleLRScheduler(lr_schedulers)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet_ensemble.train_one_epoch,
            'evaluate': resnet_ensemble.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'resnet18_sngp':
        model = resnet_sngp.resnet18_sngp(
            num_classes=10,
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.coeff,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=False,
            scale_random_features=args.model.gp.scale_random_features,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet_sngp.train_one_epoch,
            'evaluate': resnet_sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'wideresnet2810_deterministic':
        model = wideresnet2810.WideResNet2810(args.model.dropout_rate, n_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': wideresnet2810.train_one_epoch,
            'evaluate': wideresnet2810.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'wideresnet2810_sngp':
        model = wideresnet2810_sngp.WideResNetSNGP(
            dropout_rate=args.model.dropout_rate,
            num_classes=n_classes,
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.coeff,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
            num_inducing=args.model.gp.num_inducing,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=False,
            scale_random_features=args.model.gp.scale_random_features,
            mean_field_factor=args.model.gp.mean_field_factor,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.model.lr_scheduler.step_epochs,
            gamma=args.model.lr_scheduler.gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet_sngp.train_one_epoch,
            'evaluate': resnet_sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    else:
        NotImplementedError(f'Model {args.model} not implemented.')
    return model_dict
