import torchvision
import torch.nn as nn

from . import lenet, resnet_spectral_norm


def build_backbone(args, n_classes):
    if args.model.backbone == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained=False)
        backbone.fc = nn.Linear(512, n_classes)
    elif args.model.backbone == 'resnet18_spectral_norm':
        backbone = resnet_spectral_norm.resnet18(
            num_classes=n_classes,
            spectral_normalization=args.model.use_spectral_norm,
            coeff=args.model.coeff,
            n_power_iterations=args.model.n_power_iterations,
        )
        backbone.out_features = 512
    elif args.model.backbone == 'lenet':
        backbone = lenet.LeNet(n_classes=n_classes)
    else:
        raise NotImplementedError(f'Backbone {args.model.backbone} is not implemented.')
    return backbone
