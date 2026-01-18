"""
Training Callbacks for KladML SDK.

Provides callback system for training loops:
- LoggingCallback: Structured logging to projects/<project>/<experiment>/<run_id>.log
- CheckpointCallback: Model checkpointing to models/
- EarlyStoppingCallback: Early stopping with patience
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)


class ProjectLogger(Callback):
    """
    Structured logging callback that writes to project directory.
    
    Directory structure:
        projects/
        └── <project>/
            └── <experiment>/
                └── <run_id>.log
    
    Log format: JSON lines (one JSON object per line) for easy parsing.
    
    Example:
        >>> logger = ProjectLogger("sentinella", "gluformer_v1", "abc123")
        >>> logger.log("info", "Training started", {"epochs": 100})
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        run_id: str,
        projects_dir: str = "./data/projects",
        log_format: str = "jsonl",  # "jsonl" or "text"
        console_output: bool = True,  # Also print to console
        family_name: Optional[str] = None,
    ):
        """
        Initialize project logger.
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
            run_id: Unique run identifier
            projects_dir: Base directory for projects (default: ./projects)
            log_format: "jsonl" for JSON lines, "text" for plain text
            family_name: Optional family name for hierarchy
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.projects_dir = Path(projects_dir)
        self.log_format = log_format
        self.console_output = console_output
        self.family_name = family_name
        
        # Create log directory
        # New structure: projects/<project>/[<family>/]<experiment>/<run_id>/
        path_parts = [project_name]
        if family_name:
            path_parts.append(family_name)
        path_parts.append(experiment_name)
        path_parts.append(run_id)
        
        self.log_dir = self.projects_dir.joinpath(*path_parts)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file path
        extension = ".jsonl" if log_format == "jsonl" else ".log"
        self.log_file = self.log_dir / f"training{extension}"
        
        # Initialize file
        self._file_handle = None
        self._open_file()
    
    def _open_file(self) -> None:
        """Open log file for writing."""
        self._file_handle = open(self.log_file, "a", buffering=1)  # Line buffered
    
    def _close_file(self) -> None:
        """Close log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def log(self, level: str, message: str, data: Optional[Dict] = None) -> None:
        """
        Write a log entry.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            data: Optional additional data
        """
        if not self._file_handle:
            self._open_file()
        
        timestamp = datetime.now().isoformat()
        
        if self.log_format == "jsonl":
            entry = {
                "timestamp": timestamp,
                "level": level,
                "message": message,
            }
            if data:
                entry["data"] = data
            self._file_handle.write(json.dumps(entry) + "\n")
            
            # Console output (human-readable format)
            if self.console_output and level != "debug":
                console_line = f"[{timestamp[:19]}] [{level.upper():5}] {message}"
                if data:
                    # Show only key metrics, not full data dict
                    summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                        for k, v in data.items() if k in ['loss', 'train_loss', 'val_loss', 'phase', 'epoch'])
                    if summary:
                        console_line += f" | {summary}"
                print(console_line, flush=True)
        else:
            # Plain text format
            line = f"[{timestamp}] [{level.upper()}] {message}"
            if data:
                line += f" | {data}"
            self._file_handle.write(line + "\n")
            
            # Console output
            if self.console_output and level != "debug":
                print(line, flush=True)
    
    def info(self, message: str, data: Optional[Dict] = None) -> None:
        """Log info message."""
        self.log("info", message, data)
    
    def warning(self, message: str, data: Optional[Dict] = None) -> None:
        """Log warning message."""
        self.log("warning", message, data)
    
    def error(self, message: str, data: Optional[Dict] = None) -> None:
        """Log error message."""
        self.log("error", message, data)
    
    def debug(self, message: str, data: Optional[Dict] = None) -> None:
        """Log debug message."""
        self.log("debug", message, data)
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Log training start."""
        self.info("Training started", logs)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Log training end and close file."""
        self.info("Training completed", logs)
        self._close_file()
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log epoch start."""
        self.debug(f"Epoch {epoch} started", logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log epoch end with metrics."""
        self.info(f"Epoch {epoch} completed", logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Log batch (only every N batches to avoid spam)."""
        if batch % 100 == 0:
            self.debug(f"Batch {batch}", logs)
    
    @property
    def log_path(self) -> str:
        """Get the log file path."""
        return str(self.log_file)
    
    def __del__(self):
        """Cleanup on deletion."""
        self._close_file()


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback.
    
    Stops training when monitored metric stops improving.
    
    Example:
        >>> early_stop = EarlyStoppingCallback(patience=5, metric="val_loss")
        >>> if early_stop.should_stop:
        ...     break
    """
    
    def __init__(
        self,
        patience: int = 5,
        metric: str = "val_loss",
        mode: str = "min",  # "min" or "max"
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            metric: Metric to monitor
            mode: "min" for metrics where lower is better, "max" otherwise
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check if training should stop."""
        if logs is None or self.metric not in logs:
            return
        
        current = logs[self.metric]
        
        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return
        
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.metric}={self.best_value:.4f} at epoch {self.best_epoch}"
                )


class MetricsCallback(Callback):
    """
    Callback that collects metrics history.
    
    Example:
        >>> metrics_cb = MetricsCallback()
        >>> # After training
        >>> metrics_cb.history["train_loss"]  # List of losses per epoch
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Store metrics from this epoch."""
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)
