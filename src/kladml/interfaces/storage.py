"""
Storage Interface

Abstract interface for object storage operations.
Allows Core ML code to work with any storage backend (MinIO, S3, local filesystem).
"""

from abc import ABC, abstractmethod
from typing import List


class StorageInterface(ABC):
    """
    Abstract interface for object storage operations.
    
    Implementations:
    - LocalStorage (SDK): Uses local filesystem
    - MinIOStorage (Platform): Uses MinIO/S3
    - S3Storage (Platform): Uses AWS S3
    """
    
    @abstractmethod
    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        """
        Download a file from storage to local filesystem.
        
        Args:
            bucket: Bucket/container name
            key: Object key/path in storage
            local_path: Local destination path
        """
        pass
    
    @abstractmethod
    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        """
        Upload a file from local filesystem to storage.
        
        Args:
            local_path: Local source file path
            bucket: Bucket/container name
            key: Object key/path in storage
        """
        pass
    
    @abstractmethod
    def file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in storage."""
        pass
    
    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """
        List objects in a bucket with optional prefix filter.
        
        Args:
            bucket: Bucket name
            prefix: Optional prefix to filter results
            
        Returns:
            List of object keys
        """
        pass
    
    @abstractmethod
    def delete_file(self, bucket: str, key: str) -> None:
        """Delete a file from storage."""
        pass
    
    @abstractmethod
    def get_presigned_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        """
        Generate a presigned URL for direct access.
        
        Args:
            bucket: Bucket name
            key: Object key
            expires: URL expiration in seconds
            
        Returns:
            Presigned URL string (or file:// path for local storage)
        """
        pass
