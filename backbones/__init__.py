import torchvision
import torch.nn as nn

from . import lenet, resnet_spectral_norm, wide_resnet


def build_backbone(args, n_classes):
    if args.model.backbone == 'wide_resnet_28_10':
        backbone = wide_resnet.WideResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=.3,
            num_classes=n_classes
        )
    elif args.model.backbone == 'resnet18_spectral_norm':
        backbone = resnet_spectral_norm.resnet18(
            num_classes=n_classes,
            spectral_normalization=args.model.spectral_norm.use_spectral_norm,
            coeff=args.model.spectral_norm.coeff,
            n_power_iterations=args.model.spectral_norm.n_power_iterations,
        )
        backbone.out_features = 512
    elif args.model.backbone == 'lenet':
        backbone = lenet.LeNet(n_classes=n_classes)
    else:
        raise NotImplementedError(f'Backbone {args.model.backbone} is not implemented.')
    return backbone
