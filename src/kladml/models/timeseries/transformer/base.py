
from typing import Any, Optional
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from kladml.models.timeseries.base import TimeSeriesModel




from kladml.models.mixins import TorchExportMixin

class TransformerModel(TorchExportMixin, TimeSeriesModel):
    """
    Base class for Transformer-based Time Series models.
    
    Abstracts common Transformer infrastructure:
    - Hyperparameters (d_model, n_heads, etc.)
    - Device management (CPU/GPU)
    - Optimization (AdamW, Scheduler)
    - Model persistence (Save/Load)
    - TorchScript Export (via TorchExportMixin)
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        
        # 1. Model Hyperparameters
        self.d_model = self.config.get("d_model", 512)
        self.n_heads = self.config.get("n_heads", 8)
        self.e_layers = self.config.get("e_layers", 3)
        self.d_layers = self.config.get("d_layers", 2)
        self.d_ff = self.config.get("d_ff", 2048)
        self.dropout = self.config.get("dropout", 0.1)
        self.activation = self.config.get("activation", "gelu")
        
        # 2. Training Hyperparameters
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.batch_size = self.config.get("batch_size", 64)
        self.weight_decay = self.config.get("weight_decay", 1e-5)
        
        # 3. Device Management
        self.device = self._get_device()
        
        # 4. Underlying PyTorch Module (to be initialized by subclasses)
        self.model: Optional[nn.Module] = None

    def _get_device(self) -> torch.device:
        """Determine efficient training device."""
        req_device = self.config.get("device", "auto")
        if req_device != "auto":
            return torch.device(req_device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None: return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Default AdamW optimizer formulation."""
        if self.model is None:
            raise ValueError("Model has not been initialized.")
            
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
    def save(self, path: str) -> None:
        """Standard PyTorch model saving."""
        if self.model is None:
            logger.warning("No model to save.")
            return
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        if hasattr(self, '_scaler') and self._scaler is not None:
             state_dict['scaler'] = self._scaler
             
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Standard PyTorch model loading."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Trusted load for local files
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Ensure model is initialized before loading state
        # Subclass must handle architecture init (often requires dimensions from config)
        if self.model is None:
            self.build_model() # Hook for subclasses
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'scaler' in checkpoint:
            self._scaler = checkpoint['scaler']
        
        logger.info(f"Model loaded from {path}")
        self.model.to(self.device)
        self._is_trained = True

    def build_model(self):
        """Abstract method to initialize self.model."""
        raise NotImplementedError("Subclasses must implement build_model()")
        
    def state_dict(self):
        """Delegate to inner model."""
        if self.model is None:
            return {}
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        """Delegate to inner model."""
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(state_dict)
        



