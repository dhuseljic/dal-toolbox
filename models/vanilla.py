import torch
# import torch.nn as nn

from metrics.ood import ood_auroc, entropy_fn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    model.to(device)

    # Train the epoch
    running_loss, running_corrects, n_samples = 0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        n_samples += batch_size
        running_loss += loss.item()*batch_size
        running_corrects += (outputs.argmax(-1) == targets).sum().item()
    train_stats = {'train_acc': running_corrects/n_samples, 'train_loss': running_loss/n_samples}

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, dataloader_ood, criterion, device):
    model.eval()
    model.to(device)
    test_stats = {}

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Update test stats
    test_stats.update({'test_loss': criterion(logits_id, targets_id).item()})
    test_stats.update({'test_acc': (logits_id.argmax(-1) == targets_id).float().mean().item()})

    # Forward prop out of distribution
    logits_ood = []
    for inputs, targets in dataloader_ood:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_ood.append(model(inputs))
    logits_ood = torch.cat(logits_ood, dim=0).cpu()

    # Update test stats
    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    entropy_id = entropy_fn(probas_id)
    probas_ood = logits_ood.softmax(-1)
    entropy_ood = entropy_fn(probas_ood)
    test_stats.update({'auroc': ood_auroc(entropy_id, entropy_ood)})

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_ood, _ = probas_ood.max(-1)
    test_stats.update({'auroc_net_conf': ood_auroc(1-conf_id, 1-conf_ood)})

    return test_stats