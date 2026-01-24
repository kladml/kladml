
import pytest
import os
import tempfile
from pathlib import Path
from kladml.backends.local_storage import LocalStorage

@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use resolved path to avoid symlink issues on macOS /var vs /private/var
        base_path = Path(tmpdir).resolve()
        storage = LocalStorage(base_path=str(base_path))
        yield storage, base_path

def test_storage_initialization(temp_storage):
    storage, path = temp_storage
    assert storage.base_path == path
    assert path.exists()

def test_upload_and_download_file(temp_storage):
    storage, path = temp_storage
    bucket = "test-bucket"
    
    # Create local file
    local_file = path / "source.txt"
    local_file.write_bytes(b"test data")
    
    # Upload
    key = "folder/dest.txt"
    storage.upload_file(str(local_file), bucket, key)
    
    full_path = path / bucket / key
    assert full_path.exists()
    assert full_path.read_bytes() == b"test data"
    
    # Download
    download_path = path / "downloaded.txt"
    storage.download_file(bucket, key, str(download_path))
    assert download_path.exists()
    assert download_path.read_bytes() == b"test data"

def test_list_objects(temp_storage):
    storage, path = temp_storage
    bucket = "list-test"
    
    # Setup files
    (path / bucket).mkdir(parents=True, exist_ok=True)
    (path / bucket / "a.txt").touch()
    (path / bucket / "folder").mkdir()
    (path / bucket / "folder" / "b.txt").touch()
    
    # List all
    objects = storage.list_objects(bucket)
    assert "a.txt" in objects
    assert "folder/b.txt" in objects
    
    # List with prefix
    objects_prefix = storage.list_objects(bucket, prefix="folder/")
    assert "folder/b.txt" in objects_prefix
    assert "a.txt" not in objects_prefix

def test_delete_file(temp_storage):
    storage, path = temp_storage
    bucket = "del-test"
    key = "file.txt"
    
    # Create manually to test
    full_path = path / bucket / key
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.touch()
    
    assert full_path.exists()
    storage.delete_file(bucket, key)
    assert not full_path.exists()

