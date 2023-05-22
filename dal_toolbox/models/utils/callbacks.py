import time
import datetime
import logging
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from collections import defaultdict
from dal_toolbox.utils import SmoothedValue


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
    def __init__(self, log_interval=100, delimiter=' ', use_print=False):
        super().__init__()
        self.log_interval = log_interval
        self.delimiter = delimiter
        self.use_print = use_print

        self.logger = logging.getLogger(__name__)
        self.header = f"Epoch [{0}]"
        self.meters = defaultdict(SmoothedValue)

    def _log(self, log_msg):
        if self.use_print:
            print(log_msg)
        else:
            self.logger.info(log_msg)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def on_train_start(self, trainer, pl_module) -> None:
        self._start_time_train = time.time()

    def on_train_end(self, trainer, pl_module) -> None:
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train))
        log_msg = self.delimiter.join([
            f'{self.header} Total training time: {eta}',
        ])
        self._log(log_msg)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time_train_epoch = time.time()

        self.train_step = 0
        self.header = f"Epoch [{trainer.current_epoch}]"
        self.num_batches = len(trainer.train_dataloader)
        self.space_fmt = f":{len(str(self.num_batches))}d"

        self.meters = defaultdict(SmoothedValue)
        self.meters['lr'] = SmoothedValue(window_size=1, fmt="{value:.4f}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        metrics['lr'] = trainer.optimizers[0].param_groups[0]['lr']

        for key, val in metrics.items():
            batch_size = len(batch[0])
            self.meters[key].update(val, n=batch_size)

        if self.train_step % self.log_interval == 0:
            log_msg = self.delimiter.join([
                f'{self.header}',
                ("[{0" + self.space_fmt + "}").format(batch_idx)+f"/{self.num_batches}]",
                str(self),
            ])
            self._log(log_msg)
        self.train_step += 1

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train_epoch))
        log_msg = f"{self.header} Total time: {eta}"
        self._log(log_msg)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.header = f"Epoch [{trainer.current_epoch}]"
        self._start_time_val = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_val))
        log_msg = self.delimiter.join([
            f'{self.header} Total time for validation: {eta}',
        ])
        self._log(log_msg)
