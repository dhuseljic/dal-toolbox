import hydra
import torch

from lightning import Trainer
from torch.utils.data import DataLoader
from dal_toolbox.datasets import CIFAR10
from dal_toolbox.datasets.utils import PlainTransforms
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import RandomSampling
from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.deterministic.linear import LinearModel
from dal_toolbox.metrics import Accuracy
from dal_toolbox.utils import seed_everything

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    seed_everything(42)

    dino_model_name = 'dinov2_vits14' # 21 M params - 384 dimensions
    # dino_model_name = 'dinov2_vitb14' # 86 M params
    # dino_model_name = 'dinov2_vitl14' # 300 M params
    # dino_model_name = 'dinov2_vitg14' # 1100 M params
    dino_model = torch.hub.load('facebookresearch/dinov2', dino_model_name)

    data = CIFAR10('/datasets', transforms=PlainTransforms(resize=(224, 224)))

    # TODO(dhuseljic): precompute and just load for faster debugging
    train_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.train_dataset, normalize_features=True)
    val_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.val_dataset, normalize_features=True)
    test_ds = DinoFeatureDataset(dino_model=dino_model, dataset=data.test_dataset, normalize_features=True)

    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_batch_size=32,
        predict_batch_size=512,
    )
    al_datamodule.random_init(n_samples=10)

    net = LinearModel(384, 10)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    model = DeterministicModel(net, optimizer=optimizer, lr_scheduler=lr_scheduler)

    al_strategy = RandomSampling()
    
    for i_acq in range(0, 10+1):
        if i_acq != 0:
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=10,
            )
            al_datamodule.update_annotations(indices)

        model.reset_states()
        trainer = Trainer(max_epochs=10)
        trainer.fit(model, train_dataloaders=al_datamodule)
        
        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        logits = torch.cat([pred[0] for pred in predictions])
        targets = torch.cat([pred[1] for pred in predictions])
        test_stats = {
            'accuracy': Accuracy()(logits, targets)
        }
        print(test_stats)




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