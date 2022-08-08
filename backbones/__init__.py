import torchvision
import torch.nn as nn

from . import lenet, resnet, wide_resnet, spectral_resnet, spectral_wide_resnet, dropout_resnet


def build_backbone(args, n_classes):
    if args.model.backbone == 'resnet18':
        backbone = resnet.resnet18(num_classes=n_classes)
        backbone.out_features = 512
    elif args.model.backbone == 'dropout_resnet':
        backbone = dropout_resnet.DropoutResNet(block=dropout_resnet.DropoutBasicBlock, num_blocks=[2,2,2,2], num_classes=n_classes, p_drop=args.model.dropout_rate)
    elif args.model.backbone == 'wide_resnet_28_10':
        backbone = wide_resnet.WideResNet(
            depth=28, widen_factor=10, num_classes=n_classes,
            dropout_rate=args.model.dropout_rate,
        )
    elif args.model.backbone == 'spectral_wide_resnet_28_10':
        backbone = spectral_wide_resnet.WideResNet(
            depth=28, widen_factor=10, num_classes=n_classes,
            dropout_rate=args.model.dropout_rate,
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.coeff,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
        )
        backbone.out_features = 640
    elif args.model.backbone == 'spectral_resnet18':
        backbone = spectral_resnet.spectral_resnet18(
            num_classes=n_classes,
            spectral_norm=args.model.spectral_norm.use_spectral_norm,
            norm_bound=args.model.spectral_norm.coeff,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
        )
        backbone.out_features = 512
    elif args.model.backbone == 'lenet':
        backbone = lenet.LeNet(n_classes=n_classes)
    else:
        raise NotImplementedError(f'Backbone {args.model.backbone} is not implemented.')
    return backbone