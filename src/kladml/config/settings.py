"""
KladML Settings

Pydantic settings with environment variable support.
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict, 
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource
)


class KladMLSettings(BaseSettings):
    """
    KladML configuration.
    
    All settings can be overridden via environment variables prefixed with KLADML_.
    Example: KLADML_DEVICE=cuda
    """
    
    # API Configuration
    api_url: str = "http://localhost:8001/api/v1"
    api_key: str | None = None
    debug: bool = False
    
    # Path Configuration
    database_url: str | None = Field(default=None, description="Database connection string")
    artifacts_dir: str = Field(default="./data", description="Local artifacts directory")

    
    # Compute Configuration
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    executor: str = "docker"  # docker | local
    
    # Docker Configuration
    docker_image_cpu: str = "ghcr.io/kladml/worker:cpu"
    docker_image_cuda: str = "ghcr.io/kladml/worker:cuda12"
    
    # Storage Configuration (S3-Compatible)
    storage_endpoint: str | None = None
    storage_access_key: str | None = None
    storage_secret_key: str | None = None
    storage_bucket: str = "kladml"
    
    # MLflow Configuration
    mlflow_tracking_uri: str | None = None

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def set_default_mlflow_uri(cls, v: str | None) -> str:
        if v:
            return v
        # Default to local SQLite DB: sqlite:///home/user/.kladml/kladml.db
        # Default to local SQLite DB: sqlite:///home/user/.kladml/kladml.db
        base_dir = Path.home() / ".kladml"
        base_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{base_dir}/kladml.db"
    
    @field_validator("database_url", mode="before")
    @classmethod
    def set_default_database_url(cls, v: str | None) -> str:
        if v:
            return v
        # Default to local SQLite
        # We construct it here directly to avoid circular imports
        # Logic matches old get_db_path()
        base_dir = Path.home() / ".kladml"
        base_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{base_dir}/kladml.db"

    model_config = SettingsConfigDict(
        env_prefix="KLADML_",
        env_file=".env",
        extra="ignore",
        yaml_file=["kladml.yaml", str(Path.home() / ".kladml" / "config.yaml")]
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
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
