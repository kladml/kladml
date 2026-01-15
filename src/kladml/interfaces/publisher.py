"""
Publisher Interface

Abstract interface for real-time metric publishing.
Allows Core ML code to publish training metrics without Redis dependency.
"""

from abc import ABC, abstractmethod
from typing import Optional


class PublisherInterface(ABC):
    """
    Abstract interface for real-time metric publishing.
    
    Implementations:
    - ConsolePublisher (SDK): Prints to console
    - NoOpPublisher (SDK): Does nothing (silent)
    - RedisPublisher (Platform): Uses Redis pub/sub for WebSocket
    """
    
    @abstractmethod
    def publish_metric(
        self, 
        run_id: str, 
        metric_name: str, 
        value: float, 
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Publish a training metric.
        
        Args:
            run_id: MLflow run ID or training session ID
            metric_name: Name of the metric (e.g., "loss", "accuracy")
            value: Metric value
            epoch: Optional epoch number
            step: Optional step number
        """
        pass
    
    @abstractmethod
    def publish_status(self, run_id: str, status: str, message: str = "") -> None:
        """
        Publish a status update.
        
        Args:
            run_id: Run identifier
            status: Status string (e.g., "RUNNING", "COMPLETED", "FAILED")
            message: Optional status message
        """
        pass
