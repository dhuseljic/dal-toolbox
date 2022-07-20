import torch
from metrics import generalization, ood, calibration

def get_test_stats(logits_id, targets_id, logits_ood):
    # Accuracy
    acc1, = generalization.accuracy(logits_id, targets_id, (1,))

    # NLL
    CEL = torch.nn.CrossEntropyLoss(reduction='mean')
    nll = CEL(logits_id, targets_id)

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    probas_ood = logits_ood.softmax(-1)
    entropy_id = ood.entropy_fn(probas_id)
    entropy_ood = ood.entropy_fn(probas_ood)
    entropy_auroc = ood.ood_auroc(entropy_id, entropy_ood)

    # net auroc 1 - max prob
    probas_id = logits_id.softmax(-1)
    conf_id, _ = probas_id.max(-1)
    probas_ood = logits_ood.softmax(-1)
    conf_ood, _ = probas_ood.max(-1)
    conf_auroc = ood.ood_auroc(1-conf_id, 1-conf_ood)

    # Area under Precision Curve for out of domain set
    aupr = ood.ood_aupr(1-conf_id, 1-conf_ood)

    # Expected Calibration Error
    ece_model = calibration.TopLabelCalibrationError()
    ece = ece_model(probas_id, targets_id)

    return {
        'acc1': acc1.item(),
        'nll': nll.item(),
        'auroc': entropy_auroc,
        'auroc_net_conf': conf_auroc,
        'aupr': aupr,
        'ece': ece
        }