import torch
from ...metrics import generalization, calibration, ood
from ...metrics import generalization
from ...utils import MetricLogger, SmoothedValue

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
def evaluate_bertmodel(model, dataloader, epoch, criterion, device, print_freq=25):
    model.eval()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    header = "Testing:"
    for batch in metric_logger.log_every(dataloader, print_freq, header):
        batch = batch.to(device)
        targets = batch['labels']
       
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        batch_acc, = generalization.accuracy(logits, targets)
        batch_f1 = generalization.f1_macro(logits, targets, model.num_classes, device)
        batch_acc_balanced = generalization.balanced_acc(logits, targets, device)

        metric_logger.update(loss=loss.item())
        metric_logger.meters['batch_acc'].update(batch_acc.item(), n=batch_size)
        metric_logger.meters['batch_f1'].update(batch_f1.item(), n=batch_size)
        metric_logger.meters['batch_acc_balanced'].update(batch_acc_balanced.item(), n=batch_size)

    test_stats = {f"test_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Test Loss: {test_stats['test_loss_epoch']:.4f}, \
        Test Accuracy: {test_stats['test_batch_acc_epoch']:.4f}")
    print("--"*40)
    return test_stats

# @torch.no_grad()
# def evaluate(model, dataloader_id, dataloaders_ood, criterion, device):
#     model.eval()
#     model.to(device)
# 
#     # Forward prop in distribution
#     logits_id, targets_id, = [], []
#     for inputs, targets in dataloader_id:
#         inputs, targets = inputs.to(device), targets.to(device)
#         logits_id.append(model(inputs))
#         targets_id.append(targets)
#     logits_id = torch.cat(logits_id, dim=0).cpu()
#     targets_id = torch.cat(targets_id, dim=0).cpu()
# 
#     # Confidence- and entropy-Scores of in domain set logits
#     probas_id = logits_id.softmax(-1)
#     conf_id, _ = probas_id.max(-1)
#     entropy_id = ood.entropy_fn(probas_id)
# 
#     # Model specific test loss and accuracy for in domain testset
#     acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0].item()
#     prec = generalization.avg_precision(probas_id, targets_id)
#     loss = criterion(logits_id, targets_id).item()
# 
#     # Negative Log Likelihood
#     nll = torch.nn.CrossEntropyLoss(reduction='mean')(logits_id, targets_id).item()
# 
#     # Top- and Marginal Calibration Error
#     tce = calibration.TopLabelCalibrationError()(probas_id, targets_id).item()
#     mce = calibration.MarginalCalibrationError()(probas_id, targets_id).item()
# 
#     metrics = {
#         "acc1": acc1,
#         "prec": prec,
#         "loss": loss,
#         "nll": nll,
#         "tce": tce,
#         "mce": mce
#     }
# 
#     for name, dataloader_ood in dataloaders_ood.items():
#         # Forward prop out of distribution
#         logits_ood = []
#         for inputs, targets in dataloader_ood:
#             inputs, targets = inputs.to(device), targets.to(device)
#             logits_ood.append(model(inputs))
#         logits_ood = torch.cat(logits_ood, dim=0).cpu()
# 
#         # Confidence- and entropy-Scores of out of domain logits
#         probas_ood = logits_ood.softmax(-1)
#         conf_ood, _ = probas_ood.max(-1)
#         entropy_ood = ood.entropy_fn(probas_ood)
# 
#         # Area under the Precision-Recall-Curve
#         entropy_aupr = ood.ood_aupr(entropy_id, entropy_ood)
#         conf_aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)
# 
#         # Area under the Receiver-Operator-Characteristic-Curve
#         entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)
#         conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)
# 
#         # Add to metrics
#         metrics[name+"_entropy_auroc"] = entropy_auroc
#         metrics[name+"_conf_auroc"] = conf_auroc
#         metrics[name+"_entropy_aupr"] = entropy_aupr
#         metrics[name+"_conf_aupr"] = conf_aupr
# 
#     return {f"test_{k}": v for k, v in metrics.items()}
# 