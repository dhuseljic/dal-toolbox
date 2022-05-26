import torchvision
import torch.nn as nn

from . import lenet, resnet_spectral_norm


def build_backbone(args, n_classes):
    if args.backbone == 'resnet18':
        backbone = torchvision.models.resnet18(True)
        backbone.fc = nn.Linear(512, n_classes)
    elif args.backbone == 'resnet18_spectral_norm':
        model_args = args.models.sngp
        backbone = resnet_spectral_norm.resnet18(
            spectral_normalization=(model_args.coeff != 0),
            num_classes=n_classes,
            coeff=model_args.coeff
        )
    elif args.backbone == 'lenet':
        backbone = lenet.LeNet(n_classes=n_classes)
    return backbone
