import json
import logging
import os

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dal_toolbox import datasets
from dal_toolbox import metrics
from dal_toolbox.models import deterministic
from dal_toolbox.models.utils.callbacks import MetricLogger
from dal_toolbox.utils import seed_everything, is_running_on_slurm


@hydra.main(version_base=None, config_path="./configs", config_name="supervised_learning")
def main(args):
    logger = logging.getLogger(__name__)
    logger.info('Using config: \n%s', OmegaConf.to_yaml(args))
    seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Dataset
    logger.info('Building dataset.')
    dataset = build_datasets(args)
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=args.model.train_batch_size)
    validation_dataloader = DataLoader(dataset.val_dataset, batch_size=args.model.predict_batch_size)
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=args.model.predict_batch_size)

    # Setup Model
    logger.info('Building model: %s', args.model.name)
    model = build_model(args, num_classes=dataset.num_classes, input_shape=dataset.full_train_dataset.data.shape[1:])

    callbacks = []
    if is_running_on_slurm():
        callbacks.append(MetricLogger())
    trainer = L.Trainer(
        max_epochs=args.model.num_epochs,
        enable_checkpointing=False,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        enable_progress_bar=is_running_on_slurm() is False,
        check_val_every_n_epoch=args.val_interval,
    )

    logger.info('Training..')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # Evaluate resulting model
    logger.info('Evaluation..')

    predictions = trainer.predict(model, validation_dataloader)
    logits = torch.cat([pred[0] for pred in predictions])
    targets = torch.cat([pred[1] for pred in predictions])
    val_results = {
        'accuracy': metrics.Accuracy()(logits, targets).item(),
        'nll': torch.nn.CrossEntropyLoss()(logits, targets).item(),
    }
    logger.info('Evaluation results on validation dataset: %s', val_results)

    predictions = trainer.predict(model, test_dataloader)
    logits = torch.cat([pred[0] for pred in predictions])
    targets = torch.cat([pred[1] for pred in predictions])
    test_results = {
        'accuracy': metrics.Accuracy()(logits, targets).item(),
        'nll': torch.nn.CrossEntropyLoss()(logits, targets).item(),
    }
    logger.info('Evaluation results on test dataset: %s', test_results)

    # Saving results
    results = {"validation": val_results, "test": test_results}
    file_name = os.path.join(args.output_dir, 'results.json')
    logger.info("Saving results to %s.", file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    logger.info(f"Saving representations and results to {trainer.log_dir}.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    save_feature_dataset_and_model(model, dataset, device, path=trainer.log_dir,
                                   name=f"supervised_{args.model.name}_{args.dataset.name}_{results['test']['accuracy']:.3f}")


def build_model(args, num_classes, input_shape=None):
    if args.model.name == 'resnet18_deterministic':
        model = deterministic.resnet.ResNet18(num_classes=num_classes)
    elif args.model.name == 'spectral_resnet18_deterministic':
        model = deterministic.spectral_resnet.spectral_resnet18(num_classes=num_classes,
                                                                input_shape=input_shape[::-1],  # TODO (ynagel) Ask what order this is supposed to be in
                                                                norm_bound=args.model.norm_bound,
                                                                n_power_iterations=args.model.n_power_iterations,
                                                                spectral_norm=args.model.spectral_norm)
    else:
        raise NotImplementedError(f"Model {args.model.name} is not implemented!")

    optimizer = torch.optim.SGD(model.parameters(), **args.model.optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
    model = deterministic.DeterministicModel(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics={'train_acc': metrics.Accuracy()},
        val_metrics={'val_acc': metrics.Accuracy()},
    )
    return model


def build_datasets(args):
    if args.dataset.name == 'CIFAR10':
        data = datasets.CIFAR10(args.dataset_path)
    elif args.dataset.name == 'CIFAR100':
        data = datasets.CIFAR100(args.dataset_path)
    elif args.dataset.name == 'SVHN':
        data = datasets.SVHN(args.dataset_path)
    else:
        raise NotImplementedError(f"Dataset {args.dataset.name} is not implemented!")

    return data


def save_feature_dataset_and_model(model, dataset, device, path, name):
    train_dataloader = DataLoader(dataset.full_train_dataset_eval_transforms, batch_size=256)
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=256)

    feature_trainset = model.get_representations(dataloader=train_dataloader, device=device)
    feature_testset = model.get_representations(dataloader=test_dataloader, device=device)

    path = os.path.join(path + os.path.sep + f"{name}.pth")
    torch.save({'trainset': feature_trainset,
                'testset': feature_testset,
                'model': model.state_dict()}, path)


if __name__ == "__main__":
    main()
