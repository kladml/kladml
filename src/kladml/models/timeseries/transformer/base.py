
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import os
import logging
from kladml.models.timeseries.base import TimeSeriesModel

logger = logging.getLogger(__name__)


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Standard PyTorch model loading."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Trusted load for local files
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Ensure model is initialized before loading state
        # Subclass must handle architecture init (often requires dimensions from config)
        if self.model is None:
            self.build_model() # Hook for subclasses
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        self.model.to(self.device)

    def build_model(self):
        """Abstract method to initialize self.model."""
        raise NotImplementedError("Subclasses must implement build_model()")
        
    def _init_standard_callbacks(self, run_id: str, project_name: str, experiment_name: str, family_name: Optional[str] = None) -> None:
        """Initialize standard KladML callbacks (Logger, Checkpoint, EarlyStopping)."""
        from kladml.training.callbacks import ProjectLogger, CallbackList
        from kladml.training.checkpoint import CheckpointManager
        
        self._callbacks_list = CallbackList([])
        
        # 1. Project Logger (logs to projects/<project>/<family>/<experiment>/<run_id>)
        # If family_name is provided, it should be part of the path structure.
        # KladML standard seems to be project/family/experiment if family exists.
        
        self._project_logger = ProjectLogger(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            family_name=family_name # Assuming ProjectLogger supports this or we simply construct path
        )
        self._callbacks_list.append(self._project_logger)
        
        # 2. Checkpoint Manager
        self._checkpoint_manager = CheckpointManager(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            family_name=family_name
        )
        # No explicit callback for checkpointing in this simple loop, 
        # usually handled manually in train loop or via a CheckpointCallback if available.
        # CanBusModel manual loop calls self._checkpoint_manager.save_checkpoint directly.


