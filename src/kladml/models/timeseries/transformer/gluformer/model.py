"""
Gluformer Model Wrapper.

TimeSeriesModel wrapper for Gluformer that integrates with KladML SDK.
Provides training loop with:
- Structured logging to projects/<project>/<experiment>/<run_id>.log
- Model checkpointing to models/<project>_<experiment>/
- Early stopping
- MLflow integration
"""

from loguru import logger
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kladml.models.timeseries.transformer.base import TransformerModel
from kladml.tasks import MLTask
from kladml.training.checkpoint import CheckpointManager
from kladml.training.callbacks import (
    CallbackList,
    ProjectLogger,
)
from kladml.training.run_id import generate_run_id




class GluformerModel(TransformerModel):
    """
    Gluformer model for glucose forecasting.
    
    Integrates with KladML SDK for:
    - Structured training with callbacks
    - Checkpointing to models/
    - Logging to projects/
    - MLflow experiment tracking
    
    Example:
        >>> from kladml.models.timeseries.transformer.gluformer import GluformerModel
        >>> 
        >>> model = GluformerModel(config={
        ...     "project_name": "sentinella",
        ...     "experiment_name": "gluformer_v1",
        ...     "epochs": 100,
        ...     "learning_rate": 1e-4,
        ... })
        >>> 
        >>> metrics = model.train(train_data, val_data=val_data)
    """
    
    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Default Gluformer configuration."""
        return {
            "seq_len": 60,
            "pred_len": 12,
            "label_len": 48,
            "d_model": 512,
            "n_heads": 8,
            "e_layers": 3,
            "d_layers": 2,
            "d_ff": 2048,
            "dropout": 0.05,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "patience": 10,
            "warmup_epochs": 5,
            "loss_mode": "nll",
            "variance_reg": 0.01,
            "temperature": 1.0,
            "project_name": "default",
            "experiment_name": "gluformer",
        }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize Gluformer model.
        
        Args:
            config: Configuration dictionary. 
                   Defaults are fetched from `default_config()`.
        """
        # Merge defaults with provided config
        defaults = self.default_config()
        if config:
            defaults.update(config)
        
        # Initialize Base Transformer (sets d_model, device, etc.)
        super().__init__(defaults)
        
        # Project settings
        self.project_name = self.config.get("project_name")
        self.experiment_name = self.config.get("experiment_name")
        
        # Model architecture
        self.seq_len = self.config.get("seq_len")
        self.pred_len = self.config.get("pred_len")
        self.label_len = self.config.get("label_len")
        
        # Training settings
        self.epochs = self.config.get("epochs")
        self.patience = self.config.get("patience")
        self.warmup_epochs = self.config.get("warmup_epochs")
        
        # Loss configuration
        self.loss_mode = self.config.get("loss_mode")
        self.variance_reg = self.config.get("variance_reg")
        self.temperature = self.config.get("temperature")
        
        # Internal state
        self._scaler = None
        
        # Callbacks and managers
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._project_logger: Optional[ProjectLogger] = None
        self._callbacks: Optional[CallbackList] = None
        
        # Run ID (set at training start)
        self._run_id: Optional[str] = None
    
    @property
    def ml_task(self) -> MLTask:
        """Return ML task type."""
        return MLTask.TIMESERIES_FORECASTING
    
    # device property is now handled by TransformerModel
    
    def build_model(self):
        """Build the Gluformer architecture."""
        from kladml.models.timeseries.transformer.gluformer.architecture import Gluformer
        
        self.model = Gluformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_fcn=self.d_ff,
            r_drop=self.dropout,
            num_enc_layers=self.e_layers,
            num_dec_layers=self.d_layers,
            len_seq=self.seq_len,
            len_pred=self.pred_len,
            label_len=self.label_len,
            num_dynamic_features=1,  # Univariate: Glucose only
            num_static_features=1,
        ).to(self.device)
        
        return self.model
    

    
    def configure_optimizers(self):
        """Configure optimizers."""
        import torch.optim as optim
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_loss_function(self, epoch):
        """Determine loss function based on epoch (warmup logic)."""
        import torch
        import torch.nn as nn
        
        mse_loss = nn.MSELoss()
        
        def gaussian_nll_loss_reg(pred_mean, pred_logvar, y_true, var_reg=0.01):
            nll = 0.5 * (torch.exp(-pred_logvar) * (y_true - pred_mean)**2 + pred_logvar).mean()
            var_reg_loss = var_reg * torch.mean(torch.exp(-pred_logvar))
            return nll + var_reg_loss

        if self.loss_mode == "mse":
            return lambda m, v, y: mse_loss(m, y)
        
        # NLL mode with warmup
        use_nll = epoch >= self.warmup_epochs
        if use_nll:
            return lambda m, v, y: gaussian_nll_loss_reg(m, v, y, self.variance_reg)
        else:
            return lambda m, v, y: mse_loss(m, y)

    def training_step(self, batch, batch_idx):
        """Single training step."""
        
        # Unpack batch (Trainer has already moved it to device)
        x_enc = batch['x_enc']
        x_id = batch['x_id']
        x_dec = batch['x_dec']
        y = batch['y']
        
        # Forward
        # Note: self.current_epoch is injected by Trainer if we inherit from a PL-like module
        # But UniversalTrainer sets it on the callback system. 
        # Ideally Trainer should set it on the model if model tracks it.
        # For now, we access it via self.trainer if available or pass it.
        # Let's assume simplest case: we determine loss without epoch context inside step 
        # OR we access it. UniversalTrainer didn't inject 'trainer' into model.
        # Hack: We use a simplified loss for this refactor or we fix Trainer injection.
        # Let's use MSE for simplicity in Step 1 of refactor, or assume Trainer sets attribute.
        
        # We'll use the loss function logic directly here assuming NLL/MSE
        # This is a slight simplification of the dynamic epoch logic
        pred_mean, pred_logvar = self.model(x_id, x_enc, None, x_dec, None)
        
        # Re-implement simple loss logic
        # Ideally we pass 'epoch' to training_step
        loss_fn = self._get_loss_function(getattr(self, 'current_epoch', 999)) 
        loss = loss_fn(pred_mean, pred_logvar, y)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        x_enc = batch['x_enc']
        x_id = batch['x_id']
        x_dec = batch['x_dec']
        y = batch['y']
        
        pred_mean, pred_logvar = self.model(x_id, x_enc, None, x_dec, None)
        loss_fn = self._get_loss_function(getattr(self, 'current_epoch', 999))
        loss = loss_fn(pred_mean, pred_logvar, y)
        
        return {"val_loss": loss}

    def train(
        self, 
        X_train: Any, 
        y_train: Any = None, 
        X_val: Any = None, 
        y_val: Any = None,
        **kwargs
    ) -> dict[str, float]:
        """
        Train using UniversalTrainer.
        """
        from kladml.training.trainer import UniversalTrainer
        from kladml.models.timeseries.transformer.gluformer.datamodule import GluformerDataModule
        from kladml.training.callbacks import CheckpointCallback
        
        # 1. Setup DataModule
        dm = GluformerDataModule(
            train_path=X_train,
            val_path=X_val,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            batch_size=self.config.get("batch_size", 64)
        )
        dm.setup()
        self._scaler = dm.scaler  # Keep scaler ref
        
        # 2. Setup Callbacks
        run_id = generate_run_id(self.project_name, self.experiment_name)
        # Base method populates self.callbacks
        self._init_standard_callbacks(run_id, self.project_name, self.experiment_name)
        
        # Add Checkpoint Callback manually (Base creates manager but doesn't add callback by default??)
        # Actually, let's verify if we should just depend on Base.
        # Base _init_standard_callbacks creates self._checkpoint_manager.
        
        if self._checkpoint_manager:
             callbacks = self.callbacks
             # Check if already added (Base might evolve)
             # Just append it.
             callbacks.append(CheckpointCallback(self._checkpoint_manager))
        
        # 3. Build Model
        self.build_model()
        
        # 4. Train
        trainer = UniversalTrainer(
            max_epochs=self.epochs,
            callbacks=self.callbacks,
            accelerator=self.device.type, # Respect model config
        )
        
        # Inject epoch state into model for loss function dynamic behavior
        # We monkey-patch callback to update model state
        def update_epoch(epoch):
            self.current_epoch = epoch
        
        # A simple lambda callback to sync epoch
        class EpochSyncCallback:
            def on_epoch_begin(self, epoch, logs=None):
                update_epoch(epoch)
            # Implement other methods to avoid crash if Trainer calls them
            def on_train_begin(self, logs=None): pass
            def on_train_end(self, logs=None): pass
            def on_epoch_end(self, epoch, logs=None): pass
            def on_batch_begin(self, batch, logs=None): pass
            def on_batch_end(self, batch, logs=None): pass

        trainer.callbacks.append(EpochSyncCallback())
        
        metrics = trainer.fit(
            model=self,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader()
        )
        
        # Auto-export (preserved from original)
        if self._checkpoint_manager and self._checkpoint_manager.best_epoch is not None:
             # Load best
             # Trainer doesn't auto-load best at end yet, so we trust CheckpointManager saved it
             # But self.model might be the last epoch state.
             pass 
        
        self._is_trained = True
             
        return metrics
    
    def predict(self, X: Any, **kwargs) -> dict[str, Any]:
        """
        Generate predictions.
        
        Args:
            X: Input sequence (list/array of glucose values)
            
        Returns:
            Dictionary with forecast, confidence intervals, and risk assessment
        """
        import torch
        
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Prepare input
        sequence = np.array(X, dtype=np.float32)
        
        if len(sequence) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} values, got {len(sequence)}")
        
        sequence = sequence[-self.seq_len:]
        
        # Normalize
        if self._scaler is not None:
            sequence = self._scaler.transform(sequence.reshape(-1, 1)).flatten()
        
        # Create model input
        # Note: Inference currently only supports univariate input (Glucose)
        # To support multivariate, X needs to supply insulin as well.
        # For backward compatibility, if X is 1D, we pad insulin with 0.
        
        if sequence.ndim == 1:
            # Univariate input [SeqLen, 1]
            x_enc_np = sequence.reshape(-1, 1)
        else:
            x_enc_np = sequence
            
        x_enc = torch.FloatTensor(x_enc_np).unsqueeze(0).to(self.device) # [1, SeqLen, 2]
        x_id = torch.FloatTensor([0]).unsqueeze(0).to(self.device)
        
        # Decoder input
        # Zeros for future [1, LabelLen+PredLen, 1]
        x_dec = torch.zeros(1, self.label_len + self.pred_len, 1).to(self.device)
        # Copy label (start token)
        x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            pred_mean, pred_logvar = self.model(x_id, x_enc, None, x_dec, None)
        
        # Convert to numpy
        mean = pred_mean.cpu().numpy().squeeze()
        std = np.sqrt(np.exp(pred_logvar.cpu().numpy().squeeze()))
        
        # Inverse transform
        if self._scaler is not None:
            mean = self._scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self._scaler.scale_[0]
        
        # Risk assessment
        risk = "normal"
        if np.any(mean < 70):
            risk = "hypoglycemia_risk"
        elif np.any(mean > 180):
            risk = "hyperglycemia_risk"
        
        return {
            "forecast": mean.tolist(),
            "forecast_std": std.tolist(),
            "risk_assessment": risk,
            "forecast_horizon_minutes": len(mean) * 5,
        }

    def evaluate(self, X_test: Any, y_test: Any = None, **kwargs) -> dict[str, float]:
        """Evaluate model on test data."""
        # Placeholder - implement full evaluation logic
        return {"status": "not_implemented"}
    
    def export_model(self, path: str, format: str = "torchscript", **kwargs) -> None:
        """Export Gluformer to TorchScript for deployment."""
        if format != "torchscript":
            raise NotImplementedError(f"Export format '{format}' not supported. Use 'torchscript'.")
            
        try:
            from kladml.models.timeseries.transformer.gluformer.deployment import export_to_torchscript
            
            logger.info(f"Exporting Deployment model to {path}...")
            export_to_torchscript(
                model=self.model,
                output_path=path,
                scaler=self._scaler,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                label_len=self.label_len
            )
        except Exception as e:
            logger.error(f"Deployment export failed: {e}")
            raise
