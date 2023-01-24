from ...utils import MetricLogger
from ...metrics import generalization

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    train_stats = {}
    model.train()
    model.to(device)

    for i_member, (member, optim) in enumerate(zip(model, optimizer)):
        metric_logger = MetricLogger(delimiter=" ")
        header = f"Epoch [{epoch}] Model [{i_member}] " if epoch is not None else f"Model [{i_member}] "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = member(inputs)

            loss = criterion(outputs, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        train_stats.update({f"train_{k}_model{i_member}": meter.global_avg for k,
                           meter, in metric_logger.meters.items()})
    return train_stats
