
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from kladml.tasks import MLTask

class BaseModel(ABC):
    """
    Abstract base class for ML model architectures.
    
    All custom architectures must inherit from this class and implement
    the required abstract methods: train, predict, evaluate, save, load.
    
    Attributes:
        config (dict): Model configuration parameters.
        api_version (int): The API version this architecture implements.
    """
    
    # API version - increment when interface changes
    API_VERSION = 1
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the architecture.
        
        Args:
            config: Model configuration dictionary. Keys depend on the specific model.
        """
        self.config = config or {}
        self._is_trained = False
    
    @property
    @abstractmethod
    def ml_task(self) -> MLTask:
        """Required: Define which ML Task this architecture solves."""
        pass
    
    @abstractmethod
    def train(self, X_train: Any, y_train: Any = None, X_val: Any = None, y_val: Any = None, **kwargs) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (optional).
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            **kwargs: Additional training parameters.
            
        Returns:
            Dict[str, float]: Metrics dictionary (e.g., {'loss': 0.1, 'accuracy': 0.95}).
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Generate predictions.
        
        Args:
            X: Input features.
            **kwargs: Additional prediction parameters.
        
        Returns:
            Predictions (format depends on model type).
        """
        pass
        
    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on a test set.
        
        Args:
            X_test: Test data features.
            y_test: Test data labels.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model state to disk.
        
        Args:
            path: Local path where model artifact should be saved.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model state from disk.
        
        Args:
            path: Local path from where to load the model artifact.
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config.copy()
    
    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        self.config.update(params)
        return self
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    def fit(self, *args, **kwargs):
        """Alias for train()."""
        return self.train(*args, **kwargs)
        
    def export_model(self, path: str, format: str = "onnx", **kwargs) -> None:
        """
        Export the model for deployment.
        
        Args:
            path: Output path for the exported model.
            format: Export format ("onnx", "torchscript", etc.). Default: "onnx".
            **kwargs: Additional export parameters.
            
        Raises:
            NotImplementedError: If the architecture does not support export.
        """
        raise NotImplementedError(f"Export not implemented for {self.__class__.__name__}")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint (native format for resuming/fine-tuning).
        Default implementation calls save(). Override for custom behavior.
        """
        self.save(path)
    
    def run_training(self, *args, **kwargs) -> Dict[str, float]:
        """
        Full training workflow with automatic post-processing.
        
        This is the recommended entry point for training. It:
        1. Calls train() (implemented by subclass)
        2. Saves checkpoint (native format)
        3. Exports to ONNX if auto_export is enabled
        
        Returns:
            Dict with training metrics.
        """
        # 1. Run the actual training
        metrics = self.train(*args, **kwargs)
        self._is_trained = True
        
        # 2. Auto-export if configured (default: True)
        if self.config.get("auto_export", True):
            export_dir = self.config.get("export_dir", "./exports")
            export_format = self.config.get("export_format", "onnx")
            
            import os
            os.makedirs(export_dir, exist_ok=True)
            
            model_name = self.config.get("experiment_name", "model")
            export_path = os.path.join(export_dir, f"{model_name}.{export_format}")
            
            try:
                self.export_model(export_path, format=export_format)
            except NotImplementedError:
                pass  # Model doesn't support export, that's OK
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Auto-export failed: {e}")
        
        return metrics


    def _init_standard_callbacks(self, run_id: str, project_name: str, experiment_name: str) -> None:
        """
        Initialize standard training callbacks (Logging, Checkpoint, EarlyStopping).
        
        Args:
            run_id: Unique run identifier.
            project_name: Name of the project.
            experiment_name: Name of the experiment.
        """
        from kladml.training.callbacks import ProjectLogger, EarlyStoppingCallback, MetricsCallback, CallbackList
        from kladml.training.checkpoint import CheckpointManager
        
        callbacks = []
        
        # Get family name if available
        family_name = self.config.get("family_name")
        
        # 1. Project Logger
        self._project_logger = ProjectLogger(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            projects_dir="./data/projects",
            family_name=family_name,
        )
        callbacks.append(self._project_logger)
        
        # 2. Checkpoint Manager
        self._checkpoint_manager = CheckpointManager(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            base_dir="./data/projects",
            checkpoint_frequency=self.config.get("checkpoint_frequency", 5),
            family_name=family_name,
        )
        
        # 3. Early Stopping (Pluggable)
        es_nested = self.config.get("early_stopping", {})
        if isinstance(es_nested, dict) and "enabled" in es_nested:
            es_enabled = es_nested["enabled"]
        else:
            es_enabled = self.config.get("early_stopping_enabled", True)
        
        if es_enabled:
            # Patience defaults
            if isinstance(es_nested, dict) and "patience" in es_nested:
                patience = es_nested["patience"]
            else:
                patience = self.config.get("early_stopping_patience", 5)
            
            # Min Delta defaults
            if isinstance(es_nested, dict) and "min_delta" in es_nested:
                min_delta = es_nested["min_delta"]
            else:
                min_delta = self.config.get("early_stopping_min_delta", 0.0)
            
            self._early_stopping = EarlyStoppingCallback(
                patience=patience,
                metric="val_loss",
                mode="min",
                min_delta=min_delta
            )
            callbacks.append(self._early_stopping)
        else:
            self._early_stopping = None
            
        # 4. Metrics
        self._metrics_callback = MetricsCallback()
        callbacks.append(self._metrics_callback)
        
        self._callbacks_list = CallbackList(callbacks)
