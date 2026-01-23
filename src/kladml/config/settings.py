"""
KladML Settings

Pydantic settings with environment variable support.
"""

import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KladMLSettings(BaseSettings):
    """
    KladML configuration.
    
    All settings can be overridden via environment variables prefixed with KLADML_.
    Example: KLADML_DEVICE=cuda
    """
    
    # API Configuration
    api_url: str = "http://localhost:8001/api/v1"
    api_key: Optional[str] = None
    debug: bool = False

    
    # Compute Configuration
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    executor: str = "docker"  # docker | local
    
    # Docker Configuration
    docker_image_cpu: str = "ghcr.io/kladml/worker:cpu"
    docker_image_cuda: str = "ghcr.io/kladml/worker:cuda12"
    
    # Storage Configuration (S3-Compatible)
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket: str = "kladml"
    
    # MLflow Configuration
    mlflow_tracking_uri: Optional[str] = None

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def set_default_mlflow_uri(cls, v: Optional[str]) -> str:
        if v:
            return v
        # Default to local SQLite DB: sqlite:///home/user/.kladml/kladml.db
        from kladml.db.session import get_db_path
        return f"sqlite:///{get_db_path()}"
    
    model_config = SettingsConfigDict(
        env_prefix="KLADML_",
        env_file=".env",
        extra="ignore"
    )


# Global settings instance
settings = KladMLSettings()


def get_device() -> str:
    """
    Determine the compute device to use.
    
    Returns:
        str: Device string (cpu, cuda, cuda:0, mps)
    """
    if settings.device != "auto":
        return settings.device
    
    # Auto-detect
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return "cuda"
    except FileNotFoundError:
        pass
    
    # Check for MPS (Apple Silicon)
    import platform
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"
    
    return "cpu"
