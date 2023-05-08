import lightning as L


class MetricsHistory(L.Callback):
    """PyTorch Lightning callback for collecting metrics across training steps."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, module):
        metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
        self.metrics.append(metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __repr__(self) -> str:
        return str(self.metrics)
