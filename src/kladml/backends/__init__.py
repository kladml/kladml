"""
KladML Backends

Light implementations of interfaces for standalone/local use.
"""

from kladml.backends.local_storage import LocalStorage
from kladml.backends.local_config import YamlConfig
from kladml.backends.console_publisher import ConsolePublisher, NoOpPublisher
from kladml.backends.local_tracker import LocalTracker, NoOpTracker
from kladml.backends.local_metadata import LocalMetadata
from kladml.interfaces.metadata import MetadataInterface
from kladml.interfaces import TrackerInterface

def get_metadata_backend() -> MetadataInterface:
    """Get the configured metadata backend."""
    # In the future, this could read from config to return PostgresMetadata etc.
    return LocalMetadata()

def get_tracker_backend() -> TrackerInterface:
    """Get the configured tracker backend."""
    return LocalTracker()

__all__ = [
    "LocalStorage",
    "YamlConfig",
    "ConsolePublisher",
    "NoOpPublisher",
    "LocalTracker",
    "NoOpTracker",
    "LocalMetadata",
    "get_metadata_backend",
    "get_tracker_backend",
]
