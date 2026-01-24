"""
Local Tracker Backend

MLflow-based local tracking with SQLite backend.
Implements TrackerInterface.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from kladml.interfaces import TrackerInterface
from kladml.config.settings import settings

logger = logging.getLogger(__name__)


class LocalTracker(TrackerInterface):
    """
    Local experiment tracker using MLflow with SQLite backend.
    
    All data is stored locally - no server required.
    """
    
    def __init__(self, tracking_dir: Optional[str] = None):
        """
        Initialize local tracker.
        
        Args:
            tracking_dir: Directory for MLflow tracking data (legacy file-based)
        """
        # Centralized MLflow Tracking (from settings)
        self._tracking_uri = settings.mlflow_tracking_uri
        
        # Determine tracking directory
        # If using SQLite (default), we don't need a metadata dir, but we check if user provided one.
        # If tracking_dir is None, we default to clean path inside data/ 
        if tracking_dir:
            self.tracking_dir = Path(tracking_dir).resolve()
        else:
            # Use data/tracking instead of ./mlruns
            from kladml.utils.paths import get_root_data_path
            self.tracking_dir = get_root_data_path() / "tracking" / "mlruns"

        # Artifacts location
        # On centralized setup, could be ~/.kladml/mlartifacts
        self._artifact_root = str(Path.home() / ".kladml" / "mlartifacts")
        
        self._active_run = None
        self._mlflow = None
    
    def _ensure_mlflow(self):
        """Lazy-load MLflow to avoid import overhead."""
        if self._mlflow is None:
            try:
                # Only create tracking_dir if we are NOT using a DB URI 
                # OR if we explicitly want to rely on it. 
                # Actually, MLflow might auto-create 'mlruns' if not configured.
                # But here we set tracking URI.
                
                # If the URI is a local path (starts with / or file:), ensure it exists.
                # If it's sqlite:, we ensure the directory for the db exists (likely handled by db module).
                
                # We create tracking_dir only if it's being used as the backend (file store)
                is_file_store = not self._tracking_uri.startswith("sqlite:") and not self._tracking_uri.startswith("http")
                
                if is_file_store:
                     self.tracking_dir.mkdir(parents=True, exist_ok=True)
                
                import mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                raise ImportError(
                    "MLflow is required for tracking. "
                    "Install with: pip install mlflow"
                )
        return self._mlflow

    # --- Management Methods ---

    def search_experiments(self, filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for experiments."""
        mlflow = self._ensure_mlflow()
        try:
            experiments = mlflow.search_experiments(filter_string=filter_string)
            return [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time,
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            return []

    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get experiment details by name."""
        mlflow = self._ensure_mlflow()
        try:
            exp = mlflow.get_experiment_by_name(name)
            if exp:
                return {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get experiment '{name}': {e}")
            return None

    def create_experiment(self, name: str) -> str:
        """Create a new experiment and return its ID."""
        mlflow = self._ensure_mlflow()
        # Check if exists first
        existing = mlflow.get_experiment_by_name(name)
        if existing:
            return existing.experiment_id
        return mlflow.create_experiment(name)

    def search_runs(
        self, 
        experiment_id: str, 
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for runs in an experiment."""
        mlflow = self._ensure_mlflow()
        try:
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by or ["start_time DESC"],
            )
            
            result = []
            for _, run in runs.iterrows():
                result.append({
                    "run_id": run.run_id,
                    "run_name": run.get("tags.mlflow.runName", run.run_id[:8]),
                    "status": run.status,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "metrics": {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.")},
                    "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
                })
            return result
        except Exception as e:
            # If no runs found, search_runs might return empty DF or fail depending on version
            logger.warning(f"Failed to search runs for exp '{experiment_id}': {e}")
            return []

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run details by ID."""
        mlflow = self._ensure_mlflow()
        try:
            run = mlflow.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "artifact_uri": run.info.artifact_uri,
            }
        except Exception as e:
            logger.error(f"Failed to get run '{run_id}': {e}")
            return None

    # --- Logging Methods ---
    
    def start_run(
        self, 
        experiment_name: str, 
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new tracking run, optionally with a custom run_id."""
        mlflow = self._ensure_mlflow()
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run - MLflow supports passing run_id to resume/create with specific ID
        self._active_run = mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags)
        
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
            # Filter None values
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                self._mlflow.log_params(clean_params)
    
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
            # Try to detect model type and use appropriate flavor logic
            # For this MVP, we pickle it (basic support)
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
    """
    
    def __init__(self):
        pass

    # Management stubs
    def search_experiments(self, filter_string=None): return []
    def get_experiment_by_name(self, name): return None
    def create_experiment(self, name): return "noop-exp-id"
    def search_runs(self, experiment_id, **kwargs): return []
    def get_run(self, run_id): return None

    # Logging stubs
    def start_run(self, experiment_name, run_name=None, run_id=None, tags=None): return run_id or "noop-run-id"
    def end_run(self, status="FINISHED"): pass
    def log_param(self, key, value): pass
    def log_params(self, params): pass
    def log_metric(self, key, value, step=None): pass
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, local_path, artifact_path=None): pass
    def log_model(self, model, artifact_path, **kwargs): pass
    
    @property
    def active_run_id(self): return "noop-run-id"
    def get_artifact_uri(self, artifact_path=""): return ""
