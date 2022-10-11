import torch
import torch.nn as nn

from . import resnet18, resnet18_ensemble, resnet18_mcdropout, resnet18_sngp, wide_resnet_sngp


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']
    if args.model.name == 'resnet18_deterministic':
        model = resnet18.ResNet18(n_classes)
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
            'train_one_epoch': resnet18.train_one_epoch,
            'evaluate': resnet18.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'resnet18_mcdropout':
        model = resnet18_mcdropout.DropoutResNet18(n_classes, args.model.n_passes, args.model.dropout_rate)
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
            'train_one_epoch': resnet18_mcdropout.train_one_epoch,
            'evaluate': resnet18_mcdropout.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'deep_ensemble':
        members, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mem = resnet18_ensemble.Member(n_classes)
            opt = torch.optim.SGD(
                mem.parameters(),
                lr=args.model.optimizer.lr,
                weight_decay=args.model.optimizer.weight_decay,
                momentum=args.model.optimizer.momentum,
                nesterov=True
            )
            if args.model.optimizer.lr_scheduler == 'multi_step':
                lrs = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=args.model.optimizer.lr_step_epochs,
                    gamma=args.model.optimizer.lr_gamma
                )
            elif args.model.optimizer.lr_scheduler == 'cosine':
                lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
            else:
                assert "no available lr_scheduler chosen!"
            members.append(mem)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = resnet18_ensemble.Ensemble(members)
        optimizer = resnet18_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = resnet18_ensemble.EnsembleLR(lr_schedulers)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': resnet18_ensemble.train_one_epoch,
            'evaluate': resnet18_ensemble.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'resnet18_sngp':
        model = resnet18_sngp.resnet18_sngp(
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
            'train_one_epoch': resnet18_sngp.train_one_epoch,
            'evaluate': resnet18_sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }

    elif args.model.name == 'wide_resnet_sngp':
        model = wide_resnet_sngp.WideResNetSNGP(
            depth=args.model.depth,
            widen_factor=args.model.widen_factor,
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
            'train_one_epoch': resnet18_sngp.train_one_epoch,
            'evaluate': resnet18_sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    else:
        NotImplementedError(f'Model {args.model} not implemented.')
    return model_dict
