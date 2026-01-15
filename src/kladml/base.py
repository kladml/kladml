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
