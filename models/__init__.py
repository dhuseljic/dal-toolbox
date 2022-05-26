import math
import torch
import torch.nn as nn

from backbones.resnet_spectral_norm import resnet18
from backbones import build_backbone
from . import ddu, sngp, vanilla, sghmc



def build_model(args, model_params: dict):
    n_classes = model_params['n_classes']
    backbone = build_backbone(args, n_classes=n_classes)
    if 'vanilla' in args.models:
        model_args = args.models.vanilla
        optimizer = torch.optim.SGD(
            backbone.parameters(),
            lr=model_args.lr,
            weight_decay=model_args.weight_decay,
            momentum=model_args.momentum,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=model_args.lr_step_size,
            gamma=model_args.lr_gamma
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
    elif 'sghmc' in args.models:
        model_args = args.models.sghmc
        n_samples = model_params['n_samples']
        # epsilon = model_args.epsilon # 0.1/math.sqrt(n_samples)
        # C = model_args.C # .1 / epsilon
        # B_estim = model_args.B_estim # .9*C
        # resample_each = model_args.resample_each # int(1e10)
        model = sghmc.HMCModel(
            backbone,
            n_snaphots=model_args.n_snapshots,
            warm_up_batches=model_args.warm_up_batches,
            n_total_batches=args.n_epochs * math.ceil(n_samples / args.batch_size),
        )
        optimizer = sghmc.SGHMC(
            model.parameters(),
            n_samples=n_samples,
            epsilon=model_args.epsilon,
            C=model_args.C,
            B_estim=model_args.B_estim,
            resample_each=model_args.resample_each,
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
    elif 'sngp' in args.models:
        model_args = args.models.sngp
        model = sngp.SNGP(
            backbone,
            in_features=512,
            num_inducing=model_args.num_inducing,
            num_classes=n_classes,
            kernel_scale=model_args.kernel_scale,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_args.lr,
            weight_decay=model_args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=model_args.lr_step_size,
            gamma=model_args.lr_gamma
        )
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
