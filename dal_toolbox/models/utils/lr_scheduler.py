from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CosineAnnealingLRLinearWarmup(SequentialLR):
    def __init__(self, optimizer, num_epochs, warmup_epochs, eta_min=0, warmup_decay=0.01, last_epoch=-1):
        self.num_epochs = num_epochs
        self.eta_min = eta_min

        self.warmup_epochs = warmup_epochs
        self.warmup_decay = warmup_decay

        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min)
        warmup_scheduler = LinearLR(optimizer, start_factor=warmup_decay, total_iters=warmup_epochs)

        super(CosineAnnealingLRLinearWarmup, self).__init__(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
            last_epoch=last_epoch,
        )

#TODO: Scheduler von Berti