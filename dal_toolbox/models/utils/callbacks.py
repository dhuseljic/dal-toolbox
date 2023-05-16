import time
import datetime
import logging
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from collections import defaultdict
from dal_toolbox import utils


class MetricHistory(L.Callback):
    """PyTorch Lightning callback for collecting metrics across training steps."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, module):
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        self.metrics.append(metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

    def __len__(self):
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __repr__(self) -> str:
        return str(self.metrics)

    def to_list(self):
        return self.metrics


class MetricLogger(L.Callback):
    def __init__(self, log_interval=100, delimiter=' '):
        super().__init__()
        self.log_interval = log_interval
        self.delimiter = delimiter

        self.logger = logging.getLogger(__name__)
        self.meters = defaultdict(utils.SmoothedValue)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time_train = time.time()

        # Define Logging stuff
        self.i = 0
        self.header = f"Epoch [{trainer.current_epoch + 1}]"
        self.num_batches = len(trainer.train_dataloader)
        self.space_fmt = f":{len(str(self.num_batches))}d"

        self.meters = defaultdict(utils.SmoothedValue)
        self.meters['lr'] = utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        # self.metric_logger = utils.MetricLogger(delimiter=' ')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        metrics['lr'] = trainer.optimizers[0].param_groups[0]['lr']

        for key, val in metrics.items():
            batch_size = len(batch[0])
            self.meters[key].update(val, n=batch_size)

        if self.i % self.log_interval == 0:
            log_msg = self.delimiter.join([
                f'{self.header}',
                ("[{0" + self.space_fmt + "}").format(batch_idx)+f"/{self.num_batches}]",
                str(self),
            ])
            self.logger.info(log_msg)
        self.i += 1

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train))
        log_msg = f"{self.header} Total time: {eta}"
        self.logger.info(log_msg)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        pass
        # self._start_time_val = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        pass
        # metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        # eta = datetime.timedelta(seconds=int(time.time() - self._start_time_val))
        # self.delimiter.join([
        #     f'{self.header} Validation time: {eta}',
        # ])
        # log_msg =  + '  '.join([f'{n}: {m:.4f}' for n, m in metrics.items()])
        # self.logger.info(log_msg)
