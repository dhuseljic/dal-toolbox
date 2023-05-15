import hydra
import os

from omegaconf import OmegaConf
from dal_toolbox.models.deterministic import simclr 
import torchvision
from torchvision import transforms
import lightning as L
from torch.utils.data import DataLoader
import torch

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print setup
    print(OmegaConf.to_yaml(args))

    # To be reproducable
    L.seed_everything(args.random_seed)

    # Create a Data Module
    dm = CIFAR10DataModule(
        ds_path=args.ds_path,
        train_batch_size=args.model.train_batch_size,
        val_batch_size=args.model.val_batch_size,
        n_workers=args.n_cpus,
        n_epochs=args.model.n_epochs,
        random_seed=args.random_seed
    )

    optimizer = torch.optim.AdamW
    optimizer_params = args.model.optimizer

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    lr_scheduler_params = {
        'T_max' : args.model.n_epochs,
        'eta_min' : args.model.optimizer.lr / 50
    }

    # Create a Model Module
    model = simclr.SimCLR(
        hidden_dim=args.model.hidden_dim,
        temperature=args.model.temperature,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )

    # Create a Trainer Module
    trainer = L.Trainer(
        default_root_dir=os.path.join(args.cp_path, "SimCLR"),
        accelerator="auto",
        max_epochs=args.model.n_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="max", monitor="val/loss", save_top_k=3   ),
            LearningRateMonitor("epoch"),
        ],
    )

    # Train and automatically save top 5 models based on validation accuracy
    trainer.fit(model, datamodule=dm)


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self,
                ds_path: str = "./data", 
                train_batch_size: int = 256, 
                val_batch_size: int = 128, 
                n_workers: int = 1,  
                n_epochs: int = None, 
                random_seed: int = 42
            ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.train_ds = self.build_cifar10('train_contrast', self.hparams.ds_path)
        self.val_ds = self.build_cifar10('test_contrast', self.hparams.ds_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.val_batch_size, num_workers=self.hparams.n_workers)
    
    def build_cifar10(self, split, ds_path, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.262), return_info=False):
        contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=32),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if split == 'train_contrast':
            ds = torchvision.datasets.CIFAR10(ds_path, train=True, download=True, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
        elif split == 'test_contrast':
            ds = torchvision.datasets.CIFAR10(ds_path, train=False, download=True, transform=ContrastiveTransformations(contrast_transforms, n_views=2))

        if return_info:
            ds_info = {'n_classes': 10, 'mean': mean, 'std': std}
            return ds, ds_info
        return ds


if __name__ == '__main__':
    main()