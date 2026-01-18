"""
Tracker Interface

Abstract interface for experiment tracking.
Allows Core ML code to log experiments without direct MLflow dependency.
Now includes management methods for creating and retrieving experiments/runs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class TrackerInterface(ABC):
    """
    Abstract interface for experiment tracking and management.
    
    Implementations:
    - LocalTracker (SDK): MLflow with local SQLite backend
    - MLflowTracker (Platform): MLflow with PostgreSQL + S3 backend
    """
    
    # --- Management Methods ---
    
    @abstractmethod
    def search_experiments(self, filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for experiments.
        
        Returns:
            List of experiment dictionaries (id, name, etc.)
        """
        pass
    
    @abstractmethod
    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get experiment details by name."""
        pass
    
    @abstractmethod
    def create_experiment(self, name: str) -> str:
        """Create a new experiment and return its ID."""
        pass
        
    @abstractmethod
    def search_runs(
        self, 
        experiment_id: str, 
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for runs in an experiment."""
        pass
        
    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run details by ID."""
        pass

    # --- Logging Methods ---

    @abstractmethod
    def start_run(
        self, 
        experiment_name: str, 
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new tracking run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this run
            run_id: Optional custom run ID (if provided, uses this instead of generating)
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        pass
    
    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.
        
        Args:
            status: Final status (FINISHED, FAILED, KILLED)
        """
        pass
    
    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to the local file/directory
            artifact_path: Optional subdirectory in artifact storage
        """
        pass
    
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """
        Log a model artifact.
        
        Args:
            model: Model object to log
            artifact_path: Path within artifacts
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @property
    @abstractmethod
    def active_run_id(self) -> Optional[str]:
        """Get the ID of the currently active run, or None."""
        pass
    
    @abstractmethod
    def get_artifact_uri(self, artifact_path: str = "") -> str:
        """
        Get the URI for artifacts in the current run.
        
        Args:
            artifact_path: Optional path within artifacts
            
        Returns:
            Full artifact URI
        """
        pass
