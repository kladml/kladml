
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
