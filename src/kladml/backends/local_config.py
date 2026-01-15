"""
Local Config Backend

YAML + Environment variables based configuration.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from kladml.interfaces import ConfigInterface


class YamlConfig(ConfigInterface):
    """
    YAML file + environment variables configuration.
    
    Priority (highest first):
    1. Environment variables (KLADML_*)
    2. kladml.yaml in current directory
    3. ~/.kladml/config.yaml
    4. Default values
    
    Example kladml.yaml:
        project:
          name: my-project
        training:
          device: cuda
        storage:
          artifacts_dir: ./artifacts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional explicit path to config file
        """
        self._config = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from files."""
        # Try loading in order of precedence (lowest first, so higher overwrites)
        config_locations = [
            Path.home() / ".kladml" / "config.yaml",
            Path.cwd() / "kladml.yaml",
        ]
        
        if config_path:
            config_locations.append(Path(config_path))
        
        for path in config_locations:
            if path.exists():
                try:
                    with open(path) as f:
                        loaded = yaml.safe_load(f) or {}
                        self._deep_merge(self._config, loaded)
                except Exception:
                    pass  # Silently ignore malformed files
    
    def _deep_merge(self, base: dict, override: dict) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _get_env(self, key: str) -> Optional[str]:
        """Get value from environment variable."""
        env_key = f"KLADML_{key.upper().replace('.', '_')}"
        return os.environ.get(env_key)
    
    def _get_nested(self, keys: list, default: Any = None) -> Any:
        """Get nested config value."""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Supports dot notation: "training.device"
        """
        # Environment variable takes precedence
        env_val = self._get_env(key)
        if env_val is not None:
            return env_val
        
        # Then check config file
        keys = key.split(".")
        return self._get_nested(keys, default)
    
    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking URI - defaults to local SQLite."""
        return self.get("mlflow.tracking_uri", "sqlite:///mlruns.db")
    
    @property
    def storage_endpoint(self) -> Optional[str]:
        """Storage endpoint - None for local filesystem."""
        return self.get("storage.endpoint", None)
    
    @property
    def storage_access_key(self) -> Optional[str]:
        """Storage access key - None for local filesystem."""
        return self.get("storage.access_key", None)
    
    @property
    def storage_secret_key(self) -> Optional[str]:
        """Storage secret key - None for local filesystem."""
        return self.get("storage.secret_key", None)
    
    @property
    def artifacts_dir(self) -> str:
        """Artifacts directory."""
        return self.get("storage.artifacts_dir", "./kladml_data")
    
    @property
    def device(self) -> str:
        """Compute device."""
        return self.get("training.device", "auto")
