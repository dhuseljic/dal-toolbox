import torchvision
import torch.nn as nn

from . import lenet, resnet_spectral_norm


def build_backbone(args, n_classes):
    if args.model.backbone == 'resnet18':
        backbone = torchvision.models.resnet18(True)
        backbone.fc = nn.Linear(512, n_classes)
    elif args.model.backbone == 'resnet18_spectral_norm':
        backbone = resnet_spectral_norm.resnet18(
            spectral_normalization=(args.model.coeff != 0),
            coeff=args.model.coeff,
            num_classes=n_classes,
        )
    elif args.model.backbone == 'lenet':
        backbone = lenet.LeNet(n_classes=n_classes)
    else:
        raise NotImplementedError(f'Backbone {args.model.backbone} is not implemented.')
    return backbone
