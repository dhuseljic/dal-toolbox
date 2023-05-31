import logging
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
#TODO: Depcrepateded

class BasicLogger(Logger):
    """A simple logger that can run on slurm without using TQDM."""

    def __init__(self, output_dir=None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir

    @property
    def name(self):
        return "BasicLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if 'epoch' not in metrics:
            return
        epoch = metrics.pop('epoch', None)
        log_msg = f'Epoch [{epoch}] [{step}]:  ' + '  '.join([f'{n}: {m:.4f}' for n, m in metrics.items()])
        self.logger.info(log_msg)
