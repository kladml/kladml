
import logging
from typing import Dict, Optional
from kladml.training.callbacks.base import Callback

logger = logging.getLogger(__name__)

class EarlyStoppingCallback(Callback):
    """
    Early stopping callback.
    
    Stops training when monitored metric stops improving.
    
    Example:
        >>> early_stop = EarlyStoppingCallback(patience=5, metric="val_loss")
        >>> if early_stop.should_stop:
        ...     break
    """
    
    def __init__(
        self,
        patience: int = 5,
        metric: str = "val_loss",
        mode: str = "min",  # "min" or "max"
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            metric: Metric to monitor
            mode: "min" for metrics where lower is better, "max" otherwise
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check if training should stop."""
        if logs is None or self.metric not in logs:
            return
        
        current = logs[self.metric]
        
        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return
        
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.metric}={self.best_value:.4f} at epoch {self.best_epoch}"
                )
