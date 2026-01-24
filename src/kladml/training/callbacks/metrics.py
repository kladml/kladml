
from typing import Dict, List, Optional
from kladml.training.callbacks.base import Callback

class MetricsCallback(Callback):
    """
    Callback that collects metrics history.
    
    Example:
        >>> metrics_cb = MetricsCallback()
        >>> # After training
        >>> metrics_cb.history["train_loss"]  # List of losses per epoch
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Store metrics from this epoch."""
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)
