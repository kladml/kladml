"""
Console Publisher Backend

Simple console-based implementation of PublisherInterface.
"""

from typing import Optional
try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from kladml.interfaces import PublisherInterface


class ConsolePublisher(PublisherInterface):
    """
    Console publisher for development and standalone use.
    
    Prints metrics and status updates to stdout.
    Uses rich formatting if installed, otherwise simple print.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console publisher.
        
        Args:
            verbose: If False, only print status changes (not every metric)
        """
        self.verbose = verbose
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None
    
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
        
        if HAS_RICH and self.console:
            self.console.print(
                f"  📊 [dim]{metric_name}:[/dim] [bold]{value:.4f}[/bold]{context}"
            )
        else:
            print(f"  Metric: {metric_name}={value:.4f}{context}")
    
    def publish_status(self, run_id: str, status: str, message: str = "") -> None:
        """Print status to console."""
        msg_part = f" - {message}" if message else ""
        
        if HAS_RICH and self.console:
            status_colors = {
                "RUNNING": "blue",
                "COMPLETED": "green",
                "FINISHED": "green",
                "FAILED": "red",
                "KILLED": "yellow",
            }
            color = status_colors.get(status.upper(), "white")
            self.console.print(
                f"📢 [[bold {color}]{status}[/bold {color}]]{msg_part}"
            )
        else:
            print(f"[{status}] {msg_part}")


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
