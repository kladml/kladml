
from abc import ABC
from typing import Any



class Callback(ABC):
    """Base class for training callbacks."""
    
    def set_trainer(self, trainer: Any) -> None:
        """Reference to the trainer."""
        self.trainer = trainer

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, logs: dict | None = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []
    
    
    def __getitem__(self, index: int) -> Callback:
        return self.callbacks[index]
    
    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)
    
    def on_train_end(self, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)
