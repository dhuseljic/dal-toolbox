import os
import json
import hydra
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataset
from models import build_model
from utils import write_scalar_dict, seed_everything


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(args):
    print(args)
    seed_everything(args.random_seed)
    writer = SummaryWriter(log_dir=args.output_dir)

    # Load data 
    train_ds, test_ds_id, test_dss_ood, n_classes = build_dataset(args)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    test_loader_id = DataLoader(test_ds_id, batch_size=args.test_batch_size)
    test_loaders_ood = {name:DataLoader(test_ds_ood, batch_size=args.test_batch_size) for name, test_ds_ood in test_dss_ood.items()}

    # Load model
    model_dict = build_model(args, n_samples=len(train_ds), n_classes=n_classes)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    lr_scheduler = model_dict.get('lr_scheduler')

    history_train, history_test = [], []
    for i_epoch in range(args.n_epochs):
        # Train
        train_stats = train_one_epoch(model, train_loader, **model_dict['train_kwargs'], epoch=i_epoch)
        if lr_scheduler:
            lr_scheduler.step()
        history_train.append(train_stats)
        write_scalar_dict(writer, prefix='train', dict=train_stats, global_step=i_epoch)

        # Eval
        if (i_epoch+1) % args.eval_interval == 0:
            test_stats = evaluate(model, test_loader_id, test_loaders_ood, **model_dict['eval_kwargs'])
            history_test.append(test_stats)
            write_scalar_dict(writer, prefix='test', dict=test_stats, global_step=i_epoch)
            print(f"Epoch [{i_epoch}]", train_stats, test_stats)

        # Saving checkpoint
        checkpoint = {
            "args": args,
            "model": model.state_dict(),
            "optimizer": model_dict['train_kwargs']['optimizer'].state_dict(),
            "epoch": i_epoch,
            "train_history": history_train,
            "test_history": history_test,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, f"model_{i_epoch}.pth"))
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Saving results
    fname = os.path.join(args.output_dir, 'results_final.json')
    print(f"Saving results to {fname}.")
    torch.save(checkpoint, os.path.join(args.output_dir, "model_final.pth"))
    results = {'train_history': history_train, 'test_history': history_test}
    with open(fname, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
