import math
import torch
import torch.nn as nn

from backbones.resnet_spectral_norm import resnet18
from backbones import build_backbone
from . import ddu, sngp, vanilla, sghmc



def build_model(args, **kwargs):
    n_classes = kwargs['n_classes']
    backbone = build_backbone(args, n_classes=n_classes)
    if args.model.name == 'vanilla':
        optimizer = torch.optim.SGD(
            backbone.parameters(),
            lr=args.model.lr,
            weight_decay=args.model.weight_decay,
            momentum=args.model.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.model.lr_step_size,
            gamma=args.model.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': backbone,
            'optimizer': optimizer,
            'train_one_epoch': vanilla.train_one_epoch,
            'evaluate': vanilla.evaluate,
            'lr_scheduler': None,  # lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'sghmc':
        n_samples = kwargs['n_samples']
        if args.model.auto_params:
            args.model.epsilon = 0.1/math.sqrt(n_samples)
            args.model.C = .1 / args.model.epsilon
            args.model.B_estim = .9*args.model.C
            args.model.resample_each = int(1e19)

        # Auto hparams
        model = sghmc.HMCModel(
            backbone,
            n_snaphots=args.model.n_snapshots,
            warm_up_batches=args.model.warm_up_batches,
            n_total_batches=args.n_epochs * math.ceil(n_samples / args.batch_size),
        )
        optimizer = sghmc.SGHMC(
            model.parameters(),
            n_samples=n_samples,
            epsilon=args.model.epsilon,
            C=args.model.C,
            B_estim=args.model.B_estim,
            resample_each=args.model.resample_each,
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sghmc.train_one_epoch,
            'evaluate': sghmc.evaluate,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model.name == 'sngp':
        model = sngp.SNGP(
            model=backbone,
            in_features=backbone.out_features,
            num_classes=n_classes,
            num_inducing=args.model.num_inducing,
            kernel_scale=args.model.kernel_scale,
            normalize_input=args.model.normalize_input,
            cov_momentum=args.model.cov_momentum,
            ridge_penalty=args.model.ridge_penalty,
            mean_field_factor=args.model.mean_field_factor
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.model.lr,
            weight_decay=args.model.weight_decay,
            momentum=args.model.momentum,
            nesterov=True
        )
        # TODO lr scheduler
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': sngp.train_one_epoch,
            'evaluate': sngp.evaluate,
            'lr_scheduler': None,  # lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model == 'DDU':
        model = resnet18(
            spectral_normalization=(args.coeff != 0),
            num_classes=n_classes,
            coeff=args.coeff,
        )
        model = ddu.DDUWrapper(model)
        model.n_classes = n_classes
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
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
    else:
        NotImplementedError(f'Model {args.model} not implemented.')
    return model_dict
