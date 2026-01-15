"""
Local Tracker Backend

MLflow-based local tracking with SQLite backend.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from kladml.interfaces import TrackerInterface


class LocalTracker(TrackerInterface):
    """
    Local experiment tracker using MLflow with SQLite backend.
    
    All data is stored locally - no server required.
    
    Example:
        tracker = LocalTracker("./mlruns")
        run_id = tracker.start_run("my-experiment", run_name="test-1")
        tracker.log_params({"lr": 0.001, "epochs": 10})
        tracker.log_metric("loss", 0.5, step=1)
        tracker.end_run()
    """
    
    def __init__(self, tracking_dir: str = "./mlruns"):
        """
        Initialize local tracker.
        
        Args:
            tracking_dir: Directory for MLflow tracking data
        """
        self.tracking_dir = Path(tracking_dir).resolve()
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MLflow tracking URI to local SQLite
        self._tracking_uri = f"sqlite:///{self.tracking_dir}/mlflow.db"
        self._artifact_root = str(self.tracking_dir / "artifacts")
        
        self._active_run = None
        self._mlflow = None
    
    def _ensure_mlflow(self):
        """Lazy-load MLflow to avoid import overhead."""
        if self._mlflow is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                raise ImportError(
                    "MLflow is required for tracking. "
                    "Install with: pip install mlflow"
                )
        return self._mlflow
    
    def start_run(
        self, 
        experiment_name: str, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new tracking run."""
        mlflow = self._ensure_mlflow()
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        
        return self._active_run.info.run_id
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self._mlflow and self._active_run:
            self._mlflow.end_run(status=status)
            self._active_run = None
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        if self._mlflow:
            self._mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        if self._mlflow:
            self._mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file or directory as an artifact."""
        if self._mlflow:
            self._mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """Log a model artifact."""
        if self._mlflow:
            # Try to detect model type and use appropriate flavor
            # For now, just save as a generic artifact
            import tempfile
            import pickle
            
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model, f)
                temp_path = f.name
            
            try:
                self._mlflow.log_artifact(temp_path, artifact_path)
            finally:
                os.unlink(temp_path)
    
    @property
    def active_run_id(self) -> Optional[str]:
        """Get the ID of the currently active run."""
        if self._active_run:
            return self._active_run.info.run_id
        return None
    
    def get_artifact_uri(self, artifact_path: str = "") -> str:
        """Get the URI for artifacts in the current run."""
        if self._active_run:
            base_uri = self._active_run.info.artifact_uri
            if artifact_path:
                return f"{base_uri}/{artifact_path}"
            return base_uri
        return self._artifact_root


class NoOpTracker(TrackerInterface):
    """
    No-operation tracker.
    
    Does nothing - useful when MLflow is not installed or tracking is not needed.
    All methods are no-ops that return sensible defaults.
    """
    
    def __init__(self):
        self._run_id = None
    
    def start_run(
        self, 
        experiment_name: str, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a fake run ID."""
        import uuid
        self._run_id = str(uuid.uuid4())[:8]
        return self._run_id
    
    def end_run(self, status: str = "FINISHED") -> None:
        """Do nothing."""
        pass
    
    def log_param(self, key: str, value: Any) -> None:
        """Do nothing."""
        pass
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Do nothing."""
        pass
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Do nothing."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Do nothing."""
        pass
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Do nothing."""
        pass
    
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """Do nothing."""
        pass
    
    @property
    def active_run_id(self) -> Optional[str]:
        """Return the fake run ID."""
        return self._run_id
    
    def get_artifact_uri(self, artifact_path: str = "") -> str:
        """Return a local path."""
        return f"./artifacts/{artifact_path}"
