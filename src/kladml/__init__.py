"""
KladML SDK - Build custom AI architectures and preprocessors for the KladML platform.

This package provides:
- Interfaces: Abstract contracts for services (Storage, Config, Publisher, Tracker)
- Backends: Light implementations for standalone use (LocalStorage, YamlConfig, etc.)
- Models: Base classes for ML architectures (BaseModel, TimeSeriesModel, etc.)
- Training: Experiment orchestration (ExperimentRunner)
- CLI: Command-line interface for running experiments
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Core Models
from kladml.models.base import BaseModel
from kladml.data.preprocessor import BasePreprocessor
from kladml.tasks import MLTask
from kladml.models.timeseries.base import TimeSeriesModel
from kladml.models.classification.base import ClassificationModel

# Interfaces (for implementing custom backends)
from kladml.interfaces import (
    StorageInterface,
    ConfigInterface,
    PublisherInterface,
    TrackerInterface,
)

# Backends (light implementations)
from kladml.backends import (
    LocalStorage,
    YamlConfig,
    ConsolePublisher,
    NoOpPublisher,
    LocalTracker,
)

# Training
from kladml.training import ExperimentRunner

# Validation
from kladml.validator import PackageValidator, validate_package, ValidationResult

__all__ = [
    # Version
    "__version__",
    # Models
    "BaseModel", 
    "BasePreprocessor", 
    "MLTask",
    "TimeSeriesModel",
    "ClassificationModel",
    # Interfaces
    "StorageInterface",
    "ConfigInterface",
    "PublisherInterface",
    "TrackerInterface",
    # Backends
    "LocalStorage",
    "YamlConfig",
    "ConsolePublisher",
    "NoOpPublisher",
    "LocalTracker",
    # Training
    "ExperimentRunner",
    # Validation
    "PackageValidator",
    "validate_package",
    "ValidationResult",
]

