import time
import datetime
import logging
import lightning as L
from lightning.pytorch.utilities import rank_zero_only


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
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self._start_time_train = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        epoch = trainer.current_epoch + 1

        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_train))
        log_msg = f'Epoch [{epoch}] eta: {eta}  ' + '  '.join([f'{n}: {m:.4f}' for n, m in metrics.items()])
        self.logger.info(log_msg)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self._start_time_val = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}

        eta = datetime.timedelta(seconds=int(time.time() - self._start_time_val))
        log_msg = f'Validation eta: {eta}  ' + '  '.join([f'{n}: {m:.4f}' for n, m in metrics.items()])
        self.logger.info(log_msg)
