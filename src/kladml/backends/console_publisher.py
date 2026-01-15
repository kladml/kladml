"""
Console Publisher Backend

Simple console-based implementation of PublisherInterface.
"""

from typing import Optional
from rich.console import Console

from kladml.interfaces import PublisherInterface


class ConsolePublisher(PublisherInterface):
    """
    Console publisher for development and standalone use.
    
    Prints metrics and status updates to stdout with rich formatting.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console publisher.
        
        Args:
            verbose: If False, only print status changes (not every metric)
        """
        self.verbose = verbose
        self.console = Console()
    
    def publish_metric(
        self, 
        run_id: str, 
        metric_name: str, 
        value: float,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> None:
        """Print metric to console."""
        if not self.verbose:
            return
        
        parts = []
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        if step is not None:
            parts.append(f"step={step}")
        
        context = f" ({', '.join(parts)})" if parts else ""
        self.console.print(
            f"  📊 [dim]{metric_name}:[/dim] [bold]{value:.4f}[/bold]{context}"
        )
    
    def publish_status(self, run_id: str, status: str, message: str = "") -> None:
        """Print status to console."""
        status_colors = {
            "RUNNING": "blue",
            "COMPLETED": "green",
            "FINISHED": "green",
            "FAILED": "red",
            "KILLED": "yellow",
        }
        color = status_colors.get(status.upper(), "white")
        
        msg_part = f" - {message}" if message else ""
        self.console.print(
            f"📢 [[bold {color}]{status}[/bold {color}]]{msg_part}"
        )


class NoOpPublisher(PublisherInterface):
    """
    No-operation publisher.
    
    Does nothing - useful for silent/batch training.
    """
    
    def publish_metric(
        self, 
        run_id: str, 
        metric_name: str, 
        value: float,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> None:
        """Do nothing."""
        pass
    
    def publish_status(self, run_id: str, status: str, message: str = "") -> None:
        """Do nothing."""
        pass
