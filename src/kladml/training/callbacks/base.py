
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Base class for training callbacks."""
    
    def set_trainer(self, trainer: Any) -> None:
        """Reference to the trainer."""
        self.trainer = trainer

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)
