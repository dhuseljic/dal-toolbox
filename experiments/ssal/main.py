import torch
from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import PlainTransforms
from torch.utils.data import DataLoader

def main():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    cifar = CIFAR10('/datasets', transforms=PlainTransforms(resize=(224, 224)))

    train_loader = DataLoader(cifar.train_dataset, batch_size=256, shuffle=False, num_workers=4)

    torch.set_grad_enabled(False)
    dinov2_vits14.cuda()
    features = []
    for batch in train_loader:
        features.append(dinov2_vits14(batch[0].cuda()).cpu())
    features = torch.cat(features).shape

    features





if __name__ == '__main__':
    main()