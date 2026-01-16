"""
Base classes for KladML Architectures and Preprocessors.

These abstract classes define the interface that all custom implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


from kladml.tasks import MLTask

class BaseArchitecture(ABC):
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
    
    def set_params(self, **params) -> "BaseArchitecture":
        """Set model parameters."""
        self.config.update(params)
        return self
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    # Alias fit -> train for compatibility if user prefers sklearn style, 
    # BUT train is the canonical method for the Platform.
    def fit(self, *args, **kwargs):
        """Alias for train()."""
        return self.train(*args, **kwargs)
        
    def export_model(self, path: str, format: str = "torchscript", **kwargs) -> None:
        """
        Export the model for deployment.
        
        Args:
            path: Output path for the exported model.
            format: Export format (default: "torchscript").
            **kwargs: Additional export parameters.
            
        Raises:
            NotImplementedError: If the architecture does not support export.
        """
        raise NotImplementedError(f"Export not implemented for {self.__class__.__name__}")

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
        
        # 1. Project Logger
        self._project_logger = ProjectLogger(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            projects_dir="./data/projects",
        )
        callbacks.append(self._project_logger)
        
        # 2. Checkpoint Manager
        self._checkpoint_manager = CheckpointManager(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id,
            base_dir="./data/projects",
            checkpoint_frequency=self.config.get("checkpoint_frequency", 5),
        )
        # Note: CheckpointManager is not a Callback subclass in current implementation?
        # Let's check. If it's not, we don't append it to callbacks list but use it manually.
        # However, usually we want a CheckpointCallback that wraps the manager.
        # GluformerModel uses CheckpointManager manually in train loop.
        # To standardize, we should use a CheckpointCallback if possible, or keep manual usage but standard init.
        # For now, we just init it here.
        
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


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    
    All custom preprocessors must inherit from this class and implement
    the required abstract methods: fit, transform, save, load.
    
    Preprocessors transform raw datasets into formats suitable for model training.
    """
    
    # API version - increment when interface changes
    API_VERSION = 1
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessor configuration dictionary.
        """
        self.config = config or {}
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, dataset: Any) -> None:
        """
        Fit the preprocessor to the dataset (learn statistics, vocabularies, etc.).
        
        Args:
            dataset: Input dataset (format depends on preprocessor type)
        """
        pass
    
    @abstractmethod
    def transform(self, dataset: Any) -> Any:
        """
        Transform the dataset using fitted parameters.
        
        Args:
            dataset: Input dataset
        
        Returns:
            Transformed dataset
        """
        pass
    
    def fit_transform(self, dataset: Any) -> Any:
        """
        Fit and transform in one step.
        
        Args:
            dataset: Input dataset
        
        Returns:
            Transformed dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save preprocessor state to disk.
        
        Args:
            path: Directory path where state should be saved
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load preprocessor state from disk.
        
        Args:
            path: Directory path containing saved state
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted
