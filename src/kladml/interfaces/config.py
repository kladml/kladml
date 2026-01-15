"""
Config Interface

Abstract interface for configuration access.
Allows Core ML code to work without direct dependency on Platform settings.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ConfigInterface(ABC):
    """
    Abstract interface for configuration access.
    
    Implementations:
    - YamlConfig (SDK): Reads from kladml.yaml + env vars
    - PlatformConfig (Platform): Uses Platform's settings (env vars, database)
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (e.g., "mlflow_tracking_uri")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
    
    @property
    @abstractmethod
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking server URI or local path."""
        pass
    
    @property
    @abstractmethod
    def storage_endpoint(self) -> Optional[str]:
        """Object storage endpoint (MinIO/S3). None for local storage."""
        pass
    
    @property
    @abstractmethod
    def storage_access_key(self) -> Optional[str]:
        """Object storage access key. None for local storage."""
        pass
    
    @property
    @abstractmethod
    def storage_secret_key(self) -> Optional[str]:
        """Object storage secret key. None for local storage."""
        pass
    
    @property
    @abstractmethod
    def artifacts_dir(self) -> str:
        """Base directory for artifacts storage."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Compute device: auto, cpu, cuda, cuda:0, mps."""
        pass
