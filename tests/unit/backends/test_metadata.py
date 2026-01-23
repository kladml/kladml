"""
Tests for KladML Backends
"""

import pytest
import tempfile
import os
from pathlib import Path

from kladml.backends import (
    LocalStorage,
    YamlConfig,
    ConsolePublisher,
    NoOpPublisher,
)
from kladml.interfaces import (
    StorageInterface,
    ConfigInterface,
    PublisherInterface,
)


class TestLocalStorage:
    """Test LocalStorage implementation."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        return LocalStorage(str(tmp_path / "storage"))
    
    def test_implements_interface(self, storage):
        assert isinstance(storage, StorageInterface)
    
    def test_upload_and_download(self, storage, tmp_path):
        # Create a test file
        src_file = tmp_path / "source.txt"
        src_file.write_text("Hello, KladML!")
        
        # Upload
        storage.upload_file(str(src_file), "test-bucket", "test-key.txt")
        
        # Check exists
        assert storage.file_exists("test-bucket", "test-key.txt")
        
        # Download
        dst_file = tmp_path / "downloaded.txt"
        storage.download_file("test-bucket", "test-key.txt", str(dst_file))
        
        assert dst_file.read_text() == "Hello, KladML!"
    
    def test_list_objects(self, storage, tmp_path):
        # Create and upload multiple files
        for i in range(3):
            src = tmp_path / f"file{i}.txt"
            src.write_text(f"content {i}")
            storage.upload_file(str(src), "test-bucket", f"prefix/file{i}.txt")
        
        # List all
        objects = storage.list_objects("test-bucket")
        assert len(objects) == 3
        
        # List with prefix
        objects = storage.list_objects("test-bucket", "prefix/")
        assert len(objects) == 3
    
    def test_delete_file(self, storage, tmp_path):
        src = tmp_path / "to_delete.txt"
        src.write_text("delete me")
        storage.upload_file(str(src), "bucket", "key.txt")
        
        assert storage.file_exists("bucket", "key.txt")
        storage.delete_file("bucket", "key.txt")
        assert not storage.file_exists("bucket", "key.txt")
    
    def test_get_presigned_url(self, storage):
        url = storage.get_presigned_url("bucket", "key.txt")
        assert url.startswith("file://")


class TestYamlConfig:
    """Test YamlConfig implementation."""
    
    def test_implements_interface(self):
        config = YamlConfig()
        assert isinstance(config, ConfigInterface)
    
    def test_default_values(self):
        config = YamlConfig()
        assert config.mlflow_tracking_uri == "sqlite:///mlruns.db"
        assert config.artifacts_dir == "./kladml_data"
        assert config.device == "auto"
    
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("KLADML_TRAINING_DEVICE", "cuda")
        config = YamlConfig()
        assert config.get("training.device") == "cuda"
    
    def test_yaml_file_loading(self, tmp_path, monkeypatch):
        # Create a kladml.yaml
        yaml_content = """
training:
  device: mps
storage:
  artifacts_dir: /custom/path
"""
        yaml_file = tmp_path / "kladml.yaml"
        yaml_file.write_text(yaml_content)
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        config = YamlConfig()
        assert config.device == "mps"
        assert config.artifacts_dir == "/custom/path"


class TestConsolePublisher:
    """Test ConsolePublisher implementation."""
    
    def test_implements_interface(self):
        publisher = ConsolePublisher()
        assert isinstance(publisher, PublisherInterface)
    
    def test_publish_metric(self, capsys):
        publisher = ConsolePublisher(verbose=True)
        publisher.publish_metric("run-123", "loss", 0.5, epoch=1)
        
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "0.5" in captured.out
    
    def test_publish_status(self, capsys):
        publisher = ConsolePublisher()
        publisher.publish_status("run-123", "RUNNING", "Training started")
        
        captured = capsys.readouterr()
        assert "RUNNING" in captured.out


class TestNoOpPublisher:
    """Test NoOpPublisher implementation."""
    
    def test_implements_interface(self):
        publisher = NoOpPublisher()
        assert isinstance(publisher, PublisherInterface)
    
    def test_does_nothing(self, capsys):
        publisher = NoOpPublisher()
        publisher.publish_metric("run", "metric", 1.0)
        publisher.publish_status("run", "DONE")
        
        captured = capsys.readouterr()
        assert captured.out == ""
