import math
import torch
import torch.nn as nn
import torchvision

from models.resnet_spectral_norm import resnet18
from . import ddu, sngp, vanilla, sghmc


def build_model(args, model_params: dict):
    n_classes = model_params['n_classes']
    if args.model == 'vanilla':
        model = torchvision.models.resnet18(True)
        model.fc = nn.Linear(512, n_classes)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        criterion = nn.CrossEntropyLoss()
        model_dict = {
            'model': model,
            'optimizer': optimizer,
            'train_one_epoch': vanilla.train_one_epoch,
            'evaluate': vanilla.evaluate,
            'lr_scheduler': lr_scheduler,
            'train_kwargs': dict(optimizer=optimizer, criterion=criterion, device=args.device),
            'eval_kwargs': dict(criterion=criterion, device=args.device),
        }
    elif args.model == 'sghmc':
        n_samples = model_params['n_samples']
        model = torchvision.models.resnet18(True)
        model.fc = nn.Linear(512, n_classes)
        model = sghmc.HMCModel(
            model,
            n_total_batches=args.n_epochs * math.ceil(n_samples / args.batch_size),
        )
        optimizer = sghmc.SGHMC(
            model.parameters(),
            n_samples=n_samples,
            epsilon=0.1/math.sqrt(n_samples),
            prior_precision=1,
            C=10, # TODO
            B_estim=0, # TODO
            M=1,
            resample_each=int(1e10),
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
    elif args.model == 'SNGP':
        backbone = resnet18(spectral_normalization=(args.coeff != 0), num_classes=n_classes, coeff=args.coeff)
        model = sngp.SNGP(
            backbone,
            in_features=512,
            num_inducing=1024,
            num_classes=n_classes,
            kernel_scale=5,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
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
    else:
        NotImplementedError(f'Model {args.model} not implemented.')
    return model_dict
