
from typing import Any, Dict, Optional
from kladml.training.callbacks.base import Callback

class CheckpointCallback(Callback):
    """
    Triggers CheckpointManager saving at epoch end.
    Requires trainer to have .model and .optimizer attributes.
    """
    def __init__(self, manager: Any):
        self.manager = manager
        self.trainer = None
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if not self.trainer:
            return
            
        self.manager.save_checkpoint(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            epoch=epoch,
            metrics=logs or {},
            scaler=getattr(self.trainer.model, '_scaler', None) # Valid for GluformerModel
        )
