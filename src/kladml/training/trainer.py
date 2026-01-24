
from loguru import logger
import torch
from typing import Any

from kladml.training.callbacks import Callback, CallbackList
from kladml.config.schema import TrainingConfig



class UniversalTrainer:
    """
    Universal Trainer for KladML.
    
    Handles the training loop, device placement (MPS/CUDA/CPU), 
    mixed precision (future), and callbacks.
    """
    
    def __init__(
        self,
        max_epochs: int = 10,
        callbacks: list[Callback] | None = None,
        accelerator: str = "auto",  # auto, cpu, gpu, mps
        devices: str | int = "auto",
        default_root_dir: str | None = None,
        config: TrainingConfig | dict | None = None,
    ):
        # Resolve config
        if config is None:
            # Create from explicit args
            self.config = TrainingConfig(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                default_root_dir=default_root_dir,
            )
        elif isinstance(config, dict):
            # Validate dict
            # Override with explicit args if they are not default? 
            # For simplicity: explicit args + dict merging could be complex.
            # We assume if config is passed, it takes precedence, or explicit args override it.
            # Let's say explicit args override config if provided?
            # Actually, standard pattern: explicit args are defaults, config provided overrides them.
            # But here explicit args have defaults.
            # Let's construct config from dict.
            self.config = TrainingConfig(**config)
        else:
            self.config = config

        self.callbacks = CallbackList(callbacks or [])
        
        # Expose properties for backward compatibility
        self.device = self._setup_device()
        self.current_epoch = 0
        self.global_step = 0
        
    @property
    def max_epochs(self) -> int:
        return self.config.max_epochs
        
    @property
    def accelerator(self) -> str:
        return self.config.accelerator
        
    @property
    def devices(self) -> str | int:
        return self.config.devices
        
    @property
    def default_root_dir(self) -> str | None:
        return self.config.default_root_dir
        
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
        val_dataloaders: Any | None = None
    ) -> dict[str, float]:
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

    def _validate(self, model: torch.nn.Module, dataloader: Any) -> dict[str, float]:
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
