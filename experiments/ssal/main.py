import hydra
import torch
from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import PlainTransforms
from torch.utils.data import DataLoader

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):

    # dino_model_name = 'dinov2_vits14' # 21 M params
    # dino_model_name = 'dinov2_vitb14' # 86 M params
    dino_model_name = 'dinov2_vitl14' # 300 M params
    # dino_model_name = 'dinov2_vitg14' # 1100 M params
    dino_model = torch.hub.load('facebookresearch/dinov2', dino_model_name)

    data = CIFAR10('/datasets', transforms=PlainTransforms(resize=(224, 224)))

    train_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.train_dataset, normalize_features=True)
    val_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.val_dataset, normalize_features=True)
    test_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.test_dataset, normalize_features=True)



class DinoFeatureDataset:

    def __init__(self, dino_model, dataset, normalize_features=True, device='cuda'):
        features, labels = self.get_dino_features(dino_model, dataset, device)

        if normalize_features:
            features_mean = features.mean(0)
            features_std = features.std(0) + 1e-9
            features = (features - features_mean) / features_std

        self.features = features
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @torch.no_grad()
    def get_dino_features(self, dino_model, dataset, device):
        print('Getting dino features..')
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)

        features = []
        labels = []
        dino_model.to(device)
        for batch in dataloader:
            features.append(dino_model(batch[0].to(device)).to('cpu'))
            labels.append(batch[-1])
        features = torch.cat(features)
        labels = torch.cat(labels)
        return features, labels

if __name__ == '__main__':
    main()