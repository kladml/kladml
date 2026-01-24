
import logging
import torch
import platform
import subprocess
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

from kladml.training.callbacks import Callback, CallbackList

logger = logging.getLogger(__name__)

class UniversalTrainer:
    """
    Universal Trainer for KladML.
    
    Handles the training loop, device placement (MPS/CUDA/CPU), 
    mixed precision (future), and callbacks.
    """
    
    def __init__(
        self,
        max_epochs: int = 10,
        callbacks: Optional[List[Callback]] = None,
        accelerator: str = "auto",  # auto, cpu, gpu, mps
        devices: Union[str, int] = "auto",
        default_root_dir: Optional[str] = None,
    ):
        self.max_epochs = max_epochs
        self.callbacks = CallbackList(callbacks or [])
        self.accelerator = accelerator
        self.devices = devices
        self.default_root_dir = default_root_dir
        
        self.device = self._setup_device()
        self.current_epoch = 0
        self.global_step = 0
        
    def _setup_device(self) -> torch.device:
        """Determines the hardware device to use."""
        if self.accelerator == "cpu":
            return torch.device("cpu")
            
        if self.accelerator == "gpu" or self.accelerator == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
                
        if self.accelerator == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")
                
        # Auto detection
        if torch.cuda.is_available():
            return torch.device("cuda:0")
            
        if torch.backends.mps.is_available():
            return torch.device("mps")
            
        return torch.device("cpu")

    def fit(
        self, 
        model: torch.nn.Module, 
        train_dataloaders: Any, 
        val_dataloaders: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Run the full training loop.
        """
        # 1. Setup
        self.model = model.to(self.device)
        self.optimizer = None
        
        # Inject trainer ref into callbacks
        for cb in self.callbacks.callbacks:
            if hasattr(cb, 'set_trainer'):
                cb.set_trainer(self)
                
        self.callbacks.on_train_begin()
        
        # 2. Configure Optimizer
        if hasattr(model, "configure_optimizers"):
            self.optimizer = model.configure_optimizers()
        else:
            raise AttributeError("Model must implement configure_optimizers()")
            
        optimizer = self.optimizer # Alias locally
            
        # 3. Training Loop
        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch)
                
                # --- TRAIN ---
                model.train()
                train_loss_acc = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_dataloaders):
                    self.callbacks.on_batch_begin(batch_idx)
                    
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    optimizer.zero_grad()
                    
                    # Step
                    if hasattr(model, "training_step"):
                        step_output = model.training_step(batch, batch_idx)
                        loss = step_output["loss"]
                    else:
                        # Fallback for vanilla modules (not recommended)
                        x, y = batch
                        y_hat = model(x)
                        loss = torch.nn.functional.mse_loss(y_hat, y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss_acc += loss.item()
                    num_batches += 1
                    self.global_step += 1
                    
                    self.callbacks.on_batch_end(batch_idx, {"loss": loss.item()})
                
                avg_train_loss = train_loss_acc / max(1, num_batches)
                
                # --- VALIDATE ---
                val_metrics = {}
                if val_dataloaders:
                    val_metrics = self._validate(model, val_dataloaders)
                
                # --- EPOCH END ---
                metrics = {"train_loss": avg_train_loss, **val_metrics}
                self.callbacks.on_epoch_end(epoch, metrics)
                
                logger.info(f"Epoch {epoch}: {metrics}")
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
        finally:
            self.callbacks.on_train_end()
            
        return metrics

    def _validate(self, model: torch.nn.Module, dataloader: Any) -> Dict[str, float]:
        model.eval()
        val_loss_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self._move_batch_to_device(batch)
                
                if hasattr(model, "validation_step"):
                    step_output = model.validation_step(batch, batch_idx)
                    loss = step_output.get("val_loss", step_output.get("loss"))
                else:
                    x, y = batch
                    y_hat = model(x)
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                
                val_loss_acc += loss.item()
                num_batches += 1
        
        return {"val_loss": val_loss_acc / max(1, num_batches)}
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Recursively move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self._move_batch_to_device(x) for x in batch]
        elif isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        return batch
