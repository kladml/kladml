"""
KladML Interfaces

Abstract Base Classes for platform services.
These interfaces allow the Core ML code to run without concrete Platform dependencies.
"""

from kladml.interfaces.storage import StorageInterface
from kladml.interfaces.config import ConfigInterface
from kladml.interfaces.publisher import PublisherInterface
from kladml.interfaces.tracker import TrackerInterface

__all__ = [
    "StorageInterface",
    "ConfigInterface",
    "PublisherInterface",
    "TrackerInterface",
]
