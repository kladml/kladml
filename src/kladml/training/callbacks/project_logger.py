
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

from kladml.training.callbacks.base import Callback

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
            
    def log_metrics(self, data: Dict[str, Any]) -> None:
        """Alias for logging generic metric updates (e.g. intra-epoch)."""
        # We assume 'info' level for significant metric updates
        # Optionally, we can make the message generic
        self.log("info", "Metric Update", data)
    
    @property
    def log_path(self) -> str:
        """Get the log file path."""
        return str(self.log_file)
    
    def __del__(self):
        """Cleanup on deletion."""
        self._close_file()
