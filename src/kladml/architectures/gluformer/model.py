"""
Gluformer Model Wrapper.

TimeSeriesModel wrapper for Gluformer that integrates with KladML SDK.
Provides training loop with:
- Structured logging to projects/<project>/<experiment>/<run_id>.log
- Model checkpointing to models/<project>_<experiment>/
- Early stopping
- MLflow integration
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np

from kladml.models.timeseries import TimeSeriesModel
from kladml.tasks import MLTask
from kladml.training.checkpoint import CheckpointManager
from kladml.training.callbacks import (
    CallbackList,
    ProjectLogger,
    EarlyStoppingCallback,
    MetricsCallback,
)
from kladml.training.run_id import generate_run_id, get_run_checkpoint_dir

logger = logging.getLogger(__name__)


class GluformerModel(TimeSeriesModel):
    """
    Gluformer model for glucose forecasting.
    
    Integrates with KladML SDK for:
    - Structured training with callbacks
    - Checkpointing to models/
    - Logging to projects/
    - MLflow experiment tracking
    
    Example:
        >>> from kladml.architectures.gluformer import GluformerModel
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gluformer model.
        
        Args:
            config: Configuration dictionary with keys:
                - project_name: Project name (default: "default")
                - experiment_name: Experiment name (default: "gluformer")
                - seq_len: Input sequence length (default: 60)
                - pred_len: Prediction horizon (default: 12)
                - label_len: Decoder label length (default: 48)
                - d_model: Model dimension (default: 512)
                - n_heads: Attention heads (default: 8)
                - e_layers: Encoder layers (default: 3)
                - d_layers: Decoder layers (default: 2)
                - d_ff: Feedforward dimension (default: 2048)
                - dropout: Dropout rate (default: 0.05)
                - epochs: Training epochs (default: 100)
                - batch_size: Batch size (default: 64)
                - learning_rate: Learning rate (default: 1e-4)
                - patience: Early stopping patience (default: 10)
                - warmup_epochs: MSE warmup epochs (default: 5)
                - device: Training device (default: "auto")
        """
        config = config or {}
        super().__init__(config)
        
        # Project settings
        self.project_name = config.get("project_name", "default")
        self.experiment_name = config.get("experiment_name", "gluformer")
        
        # Model architecture
        self.seq_len = config.get("seq_len", 60)
        self.pred_len = config.get("pred_len", 12)
        self.label_len = config.get("label_len", 48)
        self.d_model = config.get("d_model", 512)
        self.n_heads = config.get("n_heads", 8)
        self.e_layers = config.get("e_layers", 3)
        self.d_layers = config.get("d_layers", 2)
        self.d_ff = config.get("d_ff", 2048)
        self.dropout = config.get("dropout", 0.05)
        
        # Training settings
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.patience = config.get("patience", 10)
        self.warmup_epochs = config.get("warmup_epochs", 5)
        
        # Device
        self._device = None
        self._model = None
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
    
    @property
    def device(self):
        """Get training device."""
        if self._device is None:
            import torch
            device_config = self.config.get("device", "auto")
            
            if device_config == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(device_config)
        
        return self._device
    
    def _build_model(self):
        """Build the Gluformer architecture."""
        from kladml.architectures.gluformer.architecture import Gluformer
        
        self._model = Gluformer(
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
        
        return self._model
    

    
    def train(
        self, 
        X_train: Any, 
        y_train: Any = None, 
        X_val: Any = None, 
        y_val: Any = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the Gluformer model.
        
        Args:
            X_train: Training data path or dataset
            y_train: Not used (targets embedded in data)
            X_val: Validation data path or dataset (optional)
            y_val: Not used
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of final metrics
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import uuid
        
        # Generate run ID (sequential + timestamp)
        run_id = generate_run_id(self.project_name, self.experiment_name)
        self._run_id = run_id
        
        # Initialize callbacks (Standardized)
        self._init_standard_callbacks(run_id, self.project_name, self.experiment_name)
        self._callbacks = self._callbacks_list # Alias for compatibility
        
        # Build model
        model = self._build_model()
        
        # Log training start
        self._callbacks.on_train_begin({
            "project": self.project_name,
            "experiment": self.experiment_name,
            "run_id": run_id,
            "config": self.config,
        })
        
        self._project_logger.info(f"Device: {self.device}")
        self._project_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load data
        train_loader, val_loader, scaler = self._prepare_data(X_train, X_val)
        self._scaler = scaler
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Loss functions
        def gaussian_nll_loss(pred_mean, pred_logvar, y_true):
            return 0.5 * (torch.exp(-pred_logvar) * (y_true - pred_mean)**2 + pred_logvar).mean()
        
        mse_loss = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        metrics = {}
        
        for epoch in range(self.epochs):
            self._callbacks.on_epoch_begin(epoch, {"epoch": epoch})
            
            # Check early stopping
            if self._early_stopping and self._early_stopping.should_stop:
                self._project_logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Determine loss mode (MSE warmup -> NLL)
            use_nll = epoch >= self.warmup_epochs
            loss_fn = gaussian_nll_loss if use_nll else lambda m, v, y: mse_loss(m, y)
            phase = "NLL" if use_nll else "MSE"
            
            # Train epoch
            model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                self._callbacks.on_batch_begin(batch_idx)
                
                x_enc = batch['x_enc'].to(self.device)
                x_id = batch['x_id'].to(self.device)
                x_dec = batch['x_dec'].to(self.device)
                y = batch['y'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_mean, pred_logvar = model(x_id, x_enc, None, x_dec, None)
                loss = loss_fn(pred_mean, pred_logvar, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                callbacks.on_batch_end(batch_idx, {"loss": loss.item()})
            
            avg_train_loss = np.mean(train_losses)
            epoch_metrics = {"train_loss": avg_train_loss, "phase": phase}
            
            # Validation
            if val_loader is not None:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        x_enc = batch['x_enc'].to(self.device)
                        x_id = batch['x_id'].to(self.device)
                        x_dec = batch['x_dec'].to(self.device)
                        y = batch['y'].to(self.device)
                        
                        pred_mean, pred_logvar = model(x_id, x_enc, None, x_dec, None)
                        loss = gaussian_nll_loss(pred_mean, pred_logvar, y)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                epoch_metrics["val_loss"] = avg_val_loss
                
                # Save checkpoint
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                
                self._checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    is_best=is_best,
                    comparison_metric="val_loss",
                    scaler=self._scaler,  # Include scaler for inference
                )
            
            self._callbacks.on_epoch_end(epoch, epoch_metrics)
            
            # Log progress
            log_msg = f"Epoch {epoch+1}/{self.epochs} [{phase}] | Train: {avg_train_loss:.4f}"
            if "val_loss" in epoch_metrics:
                log_msg += f" | Val: {epoch_metrics['val_loss']:.4f}"
            self._project_logger.info(log_msg)
        
        # Training complete
        metrics = {
            "train_loss": avg_train_loss,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
        }
        
        self._callbacks.on_train_end(metrics)
        
        # Load best model
        if self._checkpoint_manager.best_epoch is not None:
            self._checkpoint_manager.load_checkpoint(
                checkpoint_type="best",
                model=model,
                device=str(self.device),
            )
            self._project_logger.info(f"Restored best model from epoch {self._checkpoint_manager.best_epoch}")
            
            # --- AUTO EXPORT FOR DEPLOYMENT ---
            try:
                deploy_path = self._checkpoint_manager.checkpoint_dir / "best_model_jit.pt"
                
                self._project_logger.info(f"Auto-exporting deployment model to {deploy_path}...")
                self.export_model(str(deploy_path), format="torchscript")
            except Exception as e:
                self._project_logger.error(f"Auto-export failed: {e}")
        
        self._is_trained = True
        
        return metrics
    
    def _prepare_data(self, train_path, val_path=None):
        """
        Prepare data loaders from paths or datasets.
        
        Returns:
            Tuple of (train_loader, val_loader, scaler)
        """
        import torch
        from torch.utils.data import DataLoader
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        
        # --- HDF5 HANDLING ---
        is_hdf5 = isinstance(train_path, str) and (train_path.endswith('.h5') or train_path.endswith('.hdf5'))
        scaler = StandardScaler()

        if is_hdf5:
             # Lazy loading path
             import h5py
             with h5py.File(train_path, 'r') as f:
                 # Check for pre-computed stats in metadata
                 if 'metadata' in f and 'scaler_mean' in f['metadata'].attrs:
                     scaler.mean_ = np.array([f['metadata'].attrs['scaler_mean']])
                     scaler.scale_ = np.array([f['metadata'].attrs['scaler_scale']])
                     scaler.var_ = scaler.scale_ ** 2
                     logger.info(f"Loaded scaler stats from HDF5 metadata: mean={scaler.mean_}")
                 else:
                     logger.warning("No pre-computed scaler stats in HDF5. Fitting on first 100 series (APPROXIMATE).")
                     # Fit on subset
                     subset_values = []
                     count = 0
                     if 'series' in f:
                         # Iterate over keys (str(int))
                         for k in f['series']:
                             subset_values.extend(f['series'][k]['glucose'][:])
                             count += 1
                             if count >= 100: break
                     
                     if subset_values:
                         scaler.fit(np.array(subset_values).reshape(-1, 1))
                     else:
                         logger.warning("Could not fit scaler (dataset empty?). Inference may be unscaled.")

        else:
            # --- LEGACY PKL HANDLING ---
            # Load training data
            if isinstance(train_path, str):
                train_data = joblib.load(train_path)
            else:
                train_data = train_path
            
            # Fit scaler on full data
            all_values = []
            
            if isinstance(train_data, list):
                for item in train_data:
                    if hasattr(item, 'values'):
                        all_values.extend(item.values.flatten())
                    elif isinstance(item, dict):
                         all_values.extend(item['glucose'].flatten())
                    else:
                        all_values.extend(np.array(item).flatten())
            else:
                all_values = np.array(train_data).flatten()
            
            scaler.fit(np.array(all_values).reshape(-1, 1))

        
        # Create datasets
        from kladml.architectures.gluformer.dataset import GluformerDataset
        from kladml.data.hdf5_dataset import HDF5GluformerDataset
        
        # Check for HDF5 format
        is_hdf5 = isinstance(train_path, str) and (train_path.endswith('.h5') or train_path.endswith('.hdf5'))
        
        if is_hdf5:
            logger.info("Detected HDF5 dataset - using lazy loading")
            # For HDF5, we don't load data into RAM.
            # We assume scaler is provided or we might need a separate pass to fit it.
            # For now, we fit scaler on a subset if it's HDF5 and we don't have one?
            # Better strategy: Users should provide pre-computed scaler for massive datasets.
            # BUT to keep API compatible: we'll try to load a 'sample' from HDF5 to fit scaler 
            # or just skip fitting if user didn't pre-fit.
            # Let's check metadata for pre-computed scaler stats if we add them to conversion script.
            
            # Simple fallback: Load first 1000 series to fit scaler roughly if needed
            # Or just warn user.
            
            train_dataset = HDF5GluformerDataset(
                train_path,
                input_chunk_length=self.seq_len,
                output_chunk_length=self.pred_len,
                label_len=self.label_len,
                scaler=scaler, # Passing the fitted scaler (see below)
            )
        else:
            # Legacy/PKL loading
            train_dataset = GluformerDataset(
                train_path,
                input_chunk_length=self.seq_len,
                output_chunk_length=self.pred_len,
                label_len=self.label_len,
                scaler=scaler,
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        
        val_loader = None
        if val_path is not None:
            is_val_hdf5 = isinstance(val_path, str) and (val_path.endswith('.h5') or val_path.endswith('.hdf5'))
            
            if is_val_hdf5:
                val_dataset = HDF5GluformerDataset(
                    val_path,
                    input_chunk_length=self.seq_len,
                    output_chunk_length=self.pred_len,
                    label_len=self.label_len,
                    scaler=scaler,
                )
            else:
                val_dataset = GluformerDataset(
                    val_path,
                    input_chunk_length=self.seq_len,
                    output_chunk_length=self.pred_len,
                    label_len=self.label_len,
                    scaler=scaler,
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        return train_loader, val_loader, scaler
    
    def predict(self, X: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate predictions.
        
        Args:
            X: Input sequence (list/array of glucose values)
            
        Returns:
            Dictionary with forecast, confidence intervals, and risk assessment
        """
        import torch
        
        if self._model is None:
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
            # Add insulin channel (zeros)
            # [SeqLen, 2] -> Col 0: Glucose, Col 1: Insulin
            glucose = sequence.reshape(-1, 1)
            insulin = np.zeros_like(glucose)
            x_enc_np = np.concatenate([glucose, insulin], axis=1)
        else:
            x_enc_np = sequence
            
        x_enc = torch.FloatTensor(x_enc_np).unsqueeze(0).to(self.device) # [1, SeqLen, 2]
        x_id = torch.FloatTensor([0]).unsqueeze(0).to(self.device)
        
        # Decoder input
        # Zeros for future [1, LabelLen+PredLen, 2]
        x_dec = torch.zeros(1, self.label_len + self.pred_len, 2).to(self.device)
        # Copy label (start token)
        x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        
        # Inference
        self._model.eval()
        with torch.no_grad():
            pred_mean, pred_logvar = self._model(x_id, x_enc, None, x_dec, None)
        
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
    
    def evaluate(self, X_test: Any, y_test: Any = None, **kwargs) -> Dict[str, float]:
        """Evaluate model on test data."""
        # Placeholder - implement full evaluation logic
        return {"status": "not_implemented"}
    
    def save(self, path: str) -> None:
        """Save model to checkpoint."""
        if self._checkpoint_manager:
            logger.info(f"Model saved via CheckpointManager at: {self._checkpoint_manager.checkpoint_dir}")
        else:
            logger.warning("No checkpoint manager - model not saved")
    
    def export_model(self, path: str, format: str = "torchscript", **kwargs) -> None:
        """Export Gluformer to TorchScript for deployment."""
        if format != "torchscript":
            raise NotImplementedError(f"Export format '{format}' not supported. Use 'torchscript'.")
            
        try:
            from kladml.architectures.gluformer.deployment import export_to_torchscript
            
            logger.info(f"Exporting Deployment model to {path}...")
            export_to_torchscript(
                model=self._model,
                output_path=path,
                scaler=self._scaler,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                label_len=self.label_len
            )
        except Exception as e:
            logger.error(f"Deployment export failed: {e}")
            raise

    def load(self, path: str) -> None:
        """Load model from checkpoint."""
        import torch
        
        self._build_model()
        
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Load from checkpoint directory
            best_model = checkpoint_path / "best_model.pth"
            if best_model.exists():
                checkpoint = torch.load(best_model, map_location=self.device)
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._is_trained = True
                logger.info(f"Loaded model from {best_model}")
        else:
            # Load from specific file
            checkpoint = torch.load(path, map_location=self.device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._is_trained = True
