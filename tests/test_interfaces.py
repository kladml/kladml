"""
Tests for KladML Interfaces
"""

import pytest
from abc import ABC

from kladml.interfaces import (
    StorageInterface,
    ConfigInterface,
    PublisherInterface,
    TrackerInterface,
)


class TestStorageInterface:
    """Test StorageInterface is properly defined as ABC."""
    
    def test_is_abstract(self):
        assert issubclass(StorageInterface, ABC)
    
    def test_has_required_methods(self):
        methods = ['download_file', 'upload_file', 'file_exists', 
                   'list_objects', 'delete_file', 'get_presigned_url']
        for method in methods:
            assert hasattr(StorageInterface, method)


class TestConfigInterface:
    """Test ConfigInterface is properly defined."""
    
    def test_is_abstract(self):
        assert issubclass(ConfigInterface, ABC)
    
    def test_has_required_properties(self):
        # Check abstract properties exist
        assert hasattr(ConfigInterface, 'mlflow_tracking_uri')
        assert hasattr(ConfigInterface, 'storage_endpoint')
        assert hasattr(ConfigInterface, 'artifacts_dir')
        assert hasattr(ConfigInterface, 'device')


class TestPublisherInterface:
    """Test PublisherInterface is properly defined."""
    
    def test_is_abstract(self):
        assert issubclass(PublisherInterface, ABC)
    
    def test_has_required_methods(self):
        assert hasattr(PublisherInterface, 'publish_metric')
        assert hasattr(PublisherInterface, 'publish_status')


class TestTrackerInterface:
    """Test TrackerInterface is properly defined."""
    
    def test_is_abstract(self):
        assert issubclass(TrackerInterface, ABC)
    
    def test_has_required_methods(self):
        methods = ['start_run', 'end_run', 'log_param', 'log_params',
                   'log_metric', 'log_metrics', 'log_artifact', 'log_model']
        for method in methods:
            assert hasattr(TrackerInterface, method)
