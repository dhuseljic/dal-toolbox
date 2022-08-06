import math
import torch
import torch.nn as nn

from backbones import build_backbone
from models.mcdropout import MCDropout
from . import ddu, sngp, vanilla, sghmc, mcdropout, deep_ensemble


def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']
    backbone = build_backbone(args, n_classes=n_classes)
    if args.model.name == 'vanilla':
        optimizer = torch.optim.SGD(
            backbone.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.model.optimizer.lr_step_epochs,
            gamma=args.model.optimizer.lr_gamma
        )
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': backbone,
            'optimizer': optimizer,
            'train_one_epoch': vanilla.train_one_epoch,
            'evaluate': vanilla.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'sghmc':
        n_samples = kwargs['n_samples']
        n_total_batches = args.n_epochs * math.ceil(n_samples / args.batch_size)
        if args.model.optimizer.auto_params:
            args.model.optimizer.lr = 0.1/math.sqrt(n_samples)
            args.model.optimizer.C = .1 / args.model.optimizer.lr
            args.model.optimizer.B_estim = .8*args.model.optimizer.C
            args.model.optimizer.resample_each = int(1e19)
        model = sghmc.HMCModel(
            backbone,
            n_snaphots=args.model.ensemble.n_snapshots,
            warm_up_batches=args.model.ensemble.warm_up_batches,
            n_total_batches=n_total_batches,
        )
        optimizer = sghmc.SGHMC(
            model.parameters(),
            n_samples=n_samples,
            lr=args.model.optimizer.lr,
            C=args.model.optimizer.C,
            B_estim=args.model.optimizer.B_estim,
            resample_each=args.model.optimizer.resample_each,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.model.optimizer.lr_step_epochs,
            gamma=args.model.optimizer.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sghmc.train_one_epoch,
            'evaluate': sghmc.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'sngp':
        model = sngp.SNGP(
            model=backbone,
            in_features=backbone.out_features,
            num_inducing=args.model.gp.num_inducing,
            num_classes=n_classes,
            kernel_scale=args.model.gp.kernel_scale,
            normalize_input=args.model.gp.normalize_input,
            scale_random_features=args.model.gp.scale_random_features,
            cov_momentum=args.model.gp.cov_momentum,
            ridge_penalty=args.model.gp.ridge_penalty,
            mean_field_factor=args.model.gp.mean_field_factor
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
            milestones=args.model.optimizer.lr_step_epochs,
            gamma=args.model.optimizer.lr_gamma
        )
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sngp.train_one_epoch,
            'evaluate': sngp.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model == 'DDU':
        model = ddu.DDUWrapper(backbone)
        model.n_classes = n_classes
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': ddu.train_one_epoch,
            'evaluate': ddu.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'mcdropout':
        model = MCDropout(backbone, args.model.n_passes)
        optimizer = torch.optim.SGD(
            backbone.parameters(),
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            momentum=args.model.optimizer.momentum,
            nesterov=True
        )
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #    optimizer,
        #    milestones=args.model.optimizer.lr_step_epochs,
        #    gamma=args.model.optimizer.lr_gamma
        #)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': mcdropout.train_one_epoch,
            'evaluate': mcdropout.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'deep_ensemble':
        backbones, lr_schedulers, optimizers = [], [], []
        for _ in range(args.model.n_member):
            mod = build_backbone(args, n_classes)
            opt = torch.optim.SGD(
                    mod.parameters(),
                    lr=args.model.optimizer.lr,
                    weight_decay=args.model.optimizer.weight_decay,
                    momentum=args.model.optimizer.momentum,
                    nesterov=True
                    )
            lrs = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=args.model.optimizer.lr_step_epochs,
                    gamma=args.model.optimizer.lr_gamma)
            #lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)
            backbones.append(mod)
            optimizers.append(opt)
            lr_schedulers.append(lrs)
        model = deep_ensemble.Ensemble(backbones)
        optimizer = deep_ensemble.EnsembleOptimizer(optimizers)
        lr_scheduler = deep_ensemble.EnsembleLR(lr_schedulers)
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': deep_ensemble.train_one_epoch,
            'evaluate': deep_ensemble.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    else:
        NotImplementedError(f'Model {args.model} not implemented.')
    return model_dict