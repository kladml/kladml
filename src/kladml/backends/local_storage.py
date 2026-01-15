"""
Local Storage Backend

Filesystem-based implementation of StorageInterface.
"""

import os
import shutil
from pathlib import Path
from typing import List

from kladml.interfaces import StorageInterface


class LocalStorage(StorageInterface):
    """
    Local filesystem implementation of StorageInterface.
    
    Uses a base directory to simulate buckets as subdirectories.
    Perfect for development, testing, and standalone CLI use.
    
    Example:
        storage = LocalStorage("./artifacts")
        storage.upload_file("model.pt", "models", "my-model/v1/model.pt")
        # Creates: ./artifacts/models/my-model/v1/model.pt
    """
    
    def __init__(self, base_path: str = "./kladml_data"):
        """
        Initialize local storage.
        
        Args:
            base_path: Root directory for all storage operations
        """
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, bucket: str, key: str) -> Path:
        """Get absolute path for bucket/key combination."""
        return self.base_path / bucket / key
    
    def _ensure_parent_dirs(self, path: Path) -> None:
        """Create parent directories if they don't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        """Copy file from storage to local path."""
        src = self._get_full_path(bucket, key)
        if not src.exists():
            raise FileNotFoundError(f"File not found in storage: {bucket}/{key}")
        
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    
    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        """Copy file from local path to storage."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        dst = self._get_full_path(bucket, key)
        self._ensure_parent_dirs(dst)
        shutil.copy2(src, dst)
    
    def file_exists(self, bucket: str, key: str) -> bool:
        """Check if file exists in storage."""
        return self._get_full_path(bucket, key).exists()
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """List all objects in bucket with optional prefix filter."""
        bucket_path = self.base_path / bucket
        if not bucket_path.exists():
            return []
        
        results = []
        for item in bucket_path.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(bucket_path))
                if rel_path.startswith(prefix):
                    results.append(rel_path)
        
        return sorted(results)
    
    def delete_file(self, bucket: str, key: str) -> None:
        """Delete file from storage."""
        path = self._get_full_path(bucket, key)
        if path.exists():
            path.unlink()
    
    def get_presigned_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        """Return file:// URI for local files."""
        path = self._get_full_path(bucket, key)
        return f"file://{path}"
