"""
Custom exceptions for KladML.
"""

class KladMLError(Exception):
    """Base exception for all KladML errors."""
    pass


class ConfigurationError(KladMLError):
    """Raised when there is an issue with configuration."""
    pass


class DatasetError(KladMLError):
    """Raised when there is an issue with dataset loading or processing."""
    pass


class ModelError(KladMLError):
    """Raised when there is an issue with model creation or execution."""
    pass


class ExperimentError(KladMLError):
    """Raised when there is an issue with experiment execution."""
    pass


class StorageError(KladMLError):
    """Raised when there is an issue with storage operations."""
    pass


class RegistryError(KladMLError):
    """Raised when a component cannot be found in the registry."""
    pass
