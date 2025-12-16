import hydra
import torch
import os
import json

import mlflow
import logging
from lightning import Trainer
from torchvision import transforms
from omegaconf import OmegaConf

from dal_toolbox.datasets import CIFAR10, CustomTransforms, FeatureDataset
from dal_toolbox.models import LaplaceModel, LaplaceLinear
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import RandomSampling
from dal_toolbox import utils as dal_utils
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox import metrics

mlflow.config.enable_async_logging()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)



@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    # Seed for reproducability and print arguments to log
    dal_utils.seed_everything(args.random_seed)
    print(OmegaConf.to_yaml(args))

    # Build the dataset and initialize the al datamodule (Note that we precompute features based on DINOV2)
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    trfs = transforms.Compose([transforms.Resize((256, 256), interpolation=3), transforms.CenterCrop(224), 
                               transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    backbone_transforms = CustomTransforms(train_transform=trfs, query_transform=trfs, eval_transform=trfs)
    data = CIFAR10(args.dataset_path, transforms=backbone_transforms)
    train_ds = FeatureDataset(backbone, data.train_dataset, cache=True, cache_dir=args.dataset_path)
    test_ds = FeatureDataset(backbone, data.test_dataset, cache=True, cache_dir=args.dataset_path)
    al_datamodule = ActiveLearningDataModule(
        train_dataset=train_ds,
        query_dataset=train_ds,
        val_dataset=train_ds,
        test_dataset=test_ds,
        train_batch_size=args.model.train_batch_size,
        predict_batch_size=args.model.predict_batch_size
    )
    al_datamodule.random_init(n_samples=args.al.num_init, class_balanced=True)

    # Build the al_strategy (Here it's Random as an example)
    al_strategy = RandomSampling()

    # Build the model - in our case a laplace model for great uncertainty quantification
    num_features = len(train_ds[0][0])
    model = LaplaceLinear(
        in_features=num_features,
        out_features=data.num_classes,
        mean_field_factor=args.model.mean_field_factor,
        mc_samples=args.model.mc_samples,
        bias=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.model.num_epochs)
    model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)

    # Some config settings for the lightning trainer
    lightning_trainer_config = dict(
        max_epochs=args.model.num_epochs,
        barebones=True,
        callbacks=[MetricLogger()],
    )

    # Containers for storing metrics throughout the AL-cycles
    al_history = []

    # Start the AL cycle
    for i_acq in range(0, args.al.num_acq+1):
        # Query new samples and update the al datamodule (except for the first cycle)
        if i_acq != 0:
            indices = al_strategy.query(
                model=model,
                al_datamodule=al_datamodule,
                acq_size=args.al.acq_size,
            )
            al_datamodule.update_annotations(indices)

        # Train the model on the labeled data
        model.reset_states()
        trainer = Trainer(**lightning_trainer_config)
        trainer.fit(model, train_dataloaders=al_datamodule.train_dataloader())

        # Evaluate the models performance
        predictions = trainer.predict(model, dataloaders=al_datamodule.test_dataloader())
        test_logits, test_labels = torch.cat([pred[0] for pred in predictions]), torch.cat([pred[1] for pred in predictions])
        test_acc = metrics.Accuracy()(test_logits, test_labels).item()
        print(f'Cycle {i_acq}: Test Accuracy:', test_acc, flush=True)

        # Store evaluation metrics and al information in containers
        al_history.append({
            'test_acc' : test_acc, 
            'labeled_indices' : indices if i_acq != 0 else al_datamodule.labeled_indices
            })

    # Dump results
    with open(os.path.join(args.output_path,"results.json"), "w") as f:
        json.dump(al_history, f)
    



if __name__ == '__main__':
    main()