import torch
from ...metrics import generalization, calibration, ood
from ...metrics import generalization
from ...utils import MetricLogger


@torch.no_grad()
def evaluate_bertmodel(model, dataloader, epoch, criterion, device, print_freq=25):
    # TODO(lrauch): remove and add to trainer, maybe we need an evaluator, remove file pls
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


