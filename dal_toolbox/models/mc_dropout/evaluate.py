import torch

from ...metrics import generalization, calibration, ood


@torch.no_grad()
def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
    model.eval()
    model.to(device)

    # Get logits and targets for in-domain-test-set (Number of Samples x Number of Passes x Number of Classes)
    dropout_logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        dropout_logits_id.append(model.mc_forward(inputs, model.k))
        targets_id.append(targets)

    # Transform to tensor
    dropout_logits_id = torch.cat(dropout_logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Transform into probabilitys
    dropout_probas_id = dropout_logits_id.softmax(dim=-1)

    # Average of probas per sample
    mean_probas_id = torch.mean(dropout_probas_id, dim=1)
    mean_probas_id = ood.clamp_probas(mean_probas_id)

    # Confidence- and entropy-Scores of in domain set logits
    conf_id, _ = mean_probas_id.max(-1)
    entropy_id = ood.entropy_fn(mean_probas_id)

    # Model specific test loss and accuracy for in domain testset
    acc1 = generalization.accuracy(torch.log(mean_probas_id), targets_id, (1,))[0].item()
    prec = generalization.avg_precision(mean_probas_id, targets_id)
    loss = criterion(torch.log(mean_probas_id), targets_id).item()

    # Negative Log Likelihood
    nll = torch.nn.CrossEntropyLoss(reduction='mean')(torch.log(mean_probas_id), targets_id).item()

    # Top- and Marginal Calibration Error
    tce = calibration.TopLabelCalibrationError()(mean_probas_id, targets_id).item()
    mce = calibration.MarginalCalibrationError()(mean_probas_id, targets_id).item()

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
        dropout_logits_ood = []
        for inputs, targets in dataloader_ood:
            inputs, targets = inputs.to(device), targets.to(device)
            dropout_logits_ood.append(model.mc_forward(inputs, model.k))
        dropout_logits_ood = torch.cat(dropout_logits_ood, dim=0).cpu()
        dropout_probas_ood = dropout_logits_ood.softmax(dim=-1)
        mean_probas_ood = torch.mean(dropout_probas_ood, dim=1)
        mean_probas_ood = ood.clamp_probas(mean_probas_ood)

        # Confidence- and entropy-Scores of out of domain logits
        conf_ood, _ = mean_probas_ood.max(-1)
        entropy_ood = ood.entropy_fn(mean_probas_ood)

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
