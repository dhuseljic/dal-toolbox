import time
import datetime
import logging
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm


class MetricsHistory(L.Callback):
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


class LitProgressBar(TQDMProgressBar):
    """LitProgressBar

    Simplified bar as callback to report the training progress.
    """

    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


class Logger(L.Callback):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def on_train_epoch_start(self, trainer, module):
        self.start_time_train = time.time()

    def on_train_epoch_end(self, trainer, module):
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        epoch = trainer.current_epoch

        eta = datetime.timedelta(seconds=int(time.time() - self.start_time_train))
        log_msg = f'Epoch [{epoch}] eta: {eta}  ' + '  '.join([f'{n}: {m:.4f}' for n, m in metrics.items()])
        self.logger.info(log_msg)
