import torch
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore') 

from ...metrics import generalization, calibration, ood
from ...metrics import generalization
from ...utils import MetricLogger


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Confidence- and entropy-Scores of in domain set logits
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    entropy_id = ood.entropy_fn(probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
    prec = generalization.avg_precision(probas_id, targets_id)
    loss = criterion(logits_id, targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()

    metrics = {
        "acc1": acc1,
        "prec": prec,
        "loss": loss,
        "nll": nll,
        "tce": tce,
        "mce": mce
    }

    for name, dataloader_ood in dataloaders_ood.items():
        # Forward prop out of distribution
        logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_ood.append(model(inputs))
        logits_ood = torch.cat(logits_ood, dim=0).cpu()

        # Confidence- and entropy-Scores of out of domain logits
        probas_ood = logits_ood.softmax(-1)
        conf_ood, _ = probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(probas_ood)

        # Area under the Precision-Recall-Curve
        entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
        conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

        # Area under the Receiver-Operator-Characteristic-Curve
        entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
        conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

        # Add to metrics
        metrics[name+"_entropy_auroc"] = entropy_auroc
        metrics[name+"_conf_auroc"] = conf_auroc
        metrics[name+"_entropy_aupr"] = entropy_aupr
        metrics[name+"_conf_aupr"] = conf_aupr

    return {f"test_{k}": v for k, v in metrics.items()}

@torch.no_grad()
def evaluate_bertmodel(model, dataloader, epoch, criterion, device):
    model.eval()
    model.to(device)

    logits, targets = [],[]
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        targets.append(batch['labels'])
        logits.append(model(batch['input_ids'], batch['attention_mask']))

    logits = torch.cat(logits, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    if model.num_classes <= 2:
        test_f1_macro = generalization.f1_macro(logits, targets, 'binary')
        test_f1_micro = test_f1_macro

    else:
        test_f1_macro = generalization.f1_macro(logits, targets, 'macro')
        test_f1_micro = generalization.f1_macro(logits, targets, 'micro')

    test_stats = {
        'test_acc': generalization.accuracy(logits, targets)[0].item(),
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_acc_blc': generalization.balanced_acc(logits, targets),
        'test_loss':  criterion(logits, targets).item()
    }

    print(f"Epoch [{epoch}]: Test Loss: {test_stats['test_loss']:.4f}, \
        Test Accuracy: {test_stats['test_acc']:.4f}, \
        Test acc_blc: {test_stats['test_acc_blc']:.4f}")
    print("--"*40)
    return test_stats


