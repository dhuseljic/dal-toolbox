import os
import types

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dal_toolbox.datasets import ImageNet100
from dal_toolbox.datasets.base import BaseTransforms
from dal_toolbox.datasets.utils import FeatureDataset


class ImageNetTransform(BaseTransforms):
    def __init__(self):
        super().__init__()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),])
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                  std=(0.229, 0.224, 0.225))])  # Not sure about this

    @property
    def train_transform(self):
        return self.transforms

    @property
    def eval_transform(self):
        return self.transforms

    @property
    def query_transform(self):
        return self.transforms


def get_representations(self, dataloader, device, return_labels=True):
    with torch.no_grad():
        self.to(device)
        self.eval()
        all_features = []
        all_labels = []
        for batch in tqdm(dataloader):
            inputs = batch[0]
            labels = batch[1]
            features = self(inputs.to(device))
            all_features.append(features.cpu())
            all_labels.append(labels)
        features = torch.cat(all_features)

        if return_labels:
            labels = torch.cat(all_labels)
            return features, labels
        return features

def save_feature_dataset_and_model(model, dataset, device, path, name):
    print("Generating trainset")
    trainset = FeatureDataset(model, dataset.full_train_dataset, device)
    print("Generating testset")
    testset = FeatureDataset(model, dataset.test_dataset, device)

    path = os.path.join(path + f"{name}.pth")
    torch.save({'trainset': trainset,
                'testset': testset,
                'model': model.state_dict()}, path)
    return path

if __name__ == '__main__':
    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    data = ImageNet100("E:\ILSVRC2012", transforms=ImageNetTransform())
    train_loader = DataLoader(data.full_train_dataset_eval_transforms, batch_size=128, shuffle=False)

    for (img, label) in tqdm(train_loader):
        print(img.shape)
        feats = vits16(img)
        print(feats.shape)
        print(feats.element_size() * feats.nelement())
        break

    vits16.get_representations = types.MethodType(get_representations, vits16)
    save_feature_dataset_and_model(vits16, dataset=data, device=torch.device("cuda"), path="D:\\Dokumente\\Git\\dal-toolbox\\experiments\\active_learning\\", name="vits16_ImageNet100")
