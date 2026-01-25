
from loguru import logger
import torch
from typing import Any
from accelerate import Accelerator

from kladml.training.callbacks import Callback, CallbackList
from kladml.config.schema import TrainingConfig

class UniversalTrainer:
    """
    Universal Trainer for KladML using Hugging Face Accelerate.
    
    Handles the training loop with support for:
    - Multi-GPU / Multi-Node / MPS / TPU (via Accelerate)
    - Mixed Precision (FP16/BF16)
    - Gradient Accumulation & Clipping
    """
    
    def __init__(
        self,
        max_epochs: int = 10,
        callbacks: list[Callback] | None = None,
        accelerator: str = "auto",  # kept for config compatibility, handled by Accelerate mostly
        devices: str | int = "auto",
        default_root_dir: str | None = None,
        config: TrainingConfig | dict | None = None,
    ):
        # 1. Resolve Config
        if config is None:
            self.config = TrainingConfig(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                default_root_dir=default_root_dir,
            )
        elif isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config

        # 2. Initialize Accelerator
        # Map KladML config to Accelerator args
        mp_mode = self.config.mixed_precision
        grad_accum = self.config.gradient_accumulation_steps
        
        # Explicit CPU request override
        use_cpu = (self.config.accelerator == "cpu")
        
        self.accelerator = Accelerator(
            mixed_precision=mp_mode,
            gradient_accumulation_steps=grad_accum,
            cpu=use_cpu,
            log_with="mlflow" # We handle logging manually via Tracker usually, but this enables internal logging integration if we wanted
        )
        
        # 3. Setup State
        self.callbacks = CallbackList(callbacks or [])
        self.device = self.accelerator.device # For backward compatibility
        self.current_epoch = 0
        self.global_step = 0
        
        if self.accelerator.is_main_process:
            logger.info(f"Initialized Accelerator: {self.accelerator.state}")

    @property
    def max_epochs(self) -> int:
        return self.config.max_epochs

    def fit(
        self, 
        model: torch.nn.Module, 
        train_dataloaders: Any, 
        val_dataloaders: Any | None = None
    ) -> dict[str, float]:
        """
        Run the full training loop using Accelerate.
        """
        # 1. Prepare Model & Optimizers
        # We need to call configure_optimizers explicitly before prepare
        if hasattr(model, "configure_optimizers"):
            self.optimizer = model.configure_optimizers()
        else:
            raise AttributeError("Model must implement configure_optimizers()")
            
        optimizer = self.optimizer
        
        # Compile model (PyTorch 2.0+)
        if getattr(self.config, "compile", False): # Safely access if config has it
             if self.accelerator.device.type == "mps":
                 logger.warning("torch.compile() is currently experimental on MPS and may crash. Skipping.")
             else:
                 logger.info("Compiling model with torch.compile() for speedup...")
                 try:
                     model = torch.compile(model)
                 except Exception as e:
                     logger.warning(f"torch.compile() failed: {e}. Proceeding with eager mode.")
        
        # 2. Accelerate Prepare
        # Prepares: model (DDP wrapped), optimizer, dataloaders
        # Note: If val_dataloaders is None, we generate empty list to avoid unpack error? 
        # prepare() handles list of objects.
        
        objects_to_prepare = [model, optimizer, train_dataloaders]
        if val_dataloaders:
            objects_to_prepare.append(val_dataloaders)
            
        prepared_objects = self.accelerator.prepare(*objects_to_prepare)
        
        model = prepared_objects[0]
        optimizer = prepared_objects[1]
        train_dl = prepared_objects[2]
        val_dl = prepared_objects[3] if val_dataloaders else None
        
        self.model = model # Store wrapped model
        
        # Inject trainer ref into callbacks
        for cb in self.callbacks.callbacks:
            if hasattr(cb, 'set_trainer'):
                cb.set_trainer(self)
        
        if self.accelerator.is_main_process:
            self.callbacks.on_train_begin()
        
        # 3. Training Loop
        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                if self.accelerator.is_main_process:
                    self.callbacks.on_epoch_begin(epoch)
                
                # --- TRAIN ---
                model.train()
                train_loss_acc = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_dl):
                    if self.accelerator.is_main_process:
                        self.callbacks.on_batch_begin(batch_idx)

                    # Note: No manual .to(device), Accelerator handles it via DataLoader wrapping
                    
                    with self.accelerator.accumulate(model):
                        # Step
                        if hasattr(model, "module"): # Handle DDP wrapper for access to methods?
                            # Accelerate wraps model. If model has custom 'training_step', we might need to unwrap or call forward.
                            # Standard pattern: model(batch) or custom call.
                            # If `training_step` is defined on the LightningModule-like user model:
                            # We can try calling it. But `model` is now `DistributedDataParallel` or similar.
                            # `model.forward` typically works. 
                            # Safe bet: `loss = model(batch)` if forward returns loss? 
                            # Or unwrap for custom methods? Unwrapping removes gradients/sync logic context sometimes.
                            # Best practice: User model `forward` should return loss OR output suitable for loss.
                            # Existing KladML assumes `training_step`.
                            # DDP wrapper usually delegates method calls if they don't conflict. 
                            # But if not, we use `self.accelerator.unwrap_model(model)`? No, that's for saving.
                            pass
                        
                        # We try calling training_step directly. If wrapped, DDP in PT<1.10 didn't forward unknown methods.
                        # Accelerate wraps in standard DDP.
                        # Let's assume standard `forward` execution if `training_step` fails, or we assume `forward` IS usage.
                        # For KladML backward compact: `training_step` was used.
                        # Let's try to call it.
                        
                        try:
                            step_output = model.training_step(batch, batch_idx)
                        except AttributeError:
                             # Should handle wrapper attribute access
                             if hasattr(model, "module"):
                                  try:
                                      step_output = model.module.training_step(batch, batch_idx)
                                  except AttributeError:
                                      # Fallback to model(x)
                                      if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                          x, y = batch
                                          y_hat = model(x)
                                          loss = torch.nn.functional.mse_loss(y_hat, y)
                                          step_output = loss
                                      else:
                                          # Assume batch is input
                                          step_output = model(batch)
                             else:
                                  # Fallback to model(x)
                                  if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                      x, y = batch
                                      y_hat = model(x)
                                      loss = torch.nn.functional.mse_loss(y_hat, y)
                                      step_output = loss
                                  else:
                                      step_output = model(batch)

                        if isinstance(step_output, dict):
                            loss = step_output["loss"]
                        else:
                            loss = step_output # Assume scalar loss
                        
                        # Backpacking
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Logging (Gather loss across processes for reporting?? or just log local)
                    # For progress bars, usually local loss is fine or gather.
                    # accelerator.gather(loss) returns tensor of all losses.
                    avg_loss_batch = self.accelerator.gather(loss).mean().item()
                    
                    train_loss_acc += avg_loss_batch
                    num_batches += 1
                    self.global_step += 1
                    
                    if self.accelerator.is_main_process:
                        self.callbacks.on_batch_end(batch_idx, {"loss": avg_loss_batch})
                
                avg_train_loss = train_loss_acc / max(1, num_batches)
                
                # --- VALIDATE ---
                val_metrics = {}
                if val_dl:
                    val_metrics = self._validate(model, val_dl)
                
                # --- EPOCH END ---
                metrics = {"train_loss": avg_train_loss, **val_metrics}
                
                # Wait for everyone before logging epoch end
                self.accelerator.wait_for_everyone()
                
                if self.accelerator.is_main_process:
                    self.callbacks.on_epoch_end(epoch, metrics)
                    logger.info(f"Epoch {epoch}: {metrics}")
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
        finally:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.callbacks.on_train_end()
            
        return metrics

    def _validate(self, model: torch.nn.Module, dataloader: Any) -> dict[str, float]:
        model.eval()
        val_loss_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # No move_to_device needed, Accelerate handles it
                
                # Validation Step
                # Handle DDP wrapper
                unwrapped = self.accelerator.unwrap_model(model)
                # Actually, running validation on DDP model is fine/better for sync BN?
                # But `validation_step` access might be tricky.
                # Just assume call works via getattr or module.
                
                # Simple dispatch
                if hasattr(model, "validation_step"):
                     step_output = model.validation_step(batch, batch_idx)
                elif hasattr(model, "module") and hasattr(model.module, "validation_step"):
                     step_output = model.module.validation_step(batch, batch_idx)
                else:
                     x, y = batch
                     y_hat = model(x)
                     loss = torch.nn.functional.mse_loss(y_hat, y) # Default fallback
                     step_output = {"val_loss": loss}

                if isinstance(step_output, dict):
                   loss = step_output.get("val_loss", step_output.get("loss"))
                else:
                   loss = step_output
                
                # Gather loss from all devices to compute true average
                gathered_losses = self.accelerator.gather(loss)
                val_loss_acc += gathered_losses.mean().item()
                num_batches += 1
        
        return {"val_loss": val_loss_acc / max(1, num_batches)}

    def save_checkpoint(self, path: str):
        """Save state via Accelerate."""
        self.accelerator.save_state(path)
        
    def load_checkpoint(self, path: str):
        """Load state via Accelerate."""
        self.accelerator.load_state(path)
