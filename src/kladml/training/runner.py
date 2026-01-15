"""
Experiment Runner

Orchestrates ML experiments with dependency injection for all services.
"""

from typing import Any, Dict, Optional, Type
from pathlib import Path

from kladml.interfaces import (
    StorageInterface,
    ConfigInterface,
    PublisherInterface,
    TrackerInterface,
)
from kladml.backends import (
    LocalStorage,
    YamlConfig,
    ConsolePublisher,
    LocalTracker,
)


class ExperimentRunner:
    """
    Orchestrator for ML experiments.
    
    Uses dependency injection for all services, allowing the same code
    to run locally (standalone) or on the platform (with MinIO, Redis, etc).
    
    Example (standalone):
        runner = ExperimentRunner()  # Uses all local backends
        runner.run(model_class=MyModel, config={"epochs": 10})
    
    Example (platform):
        runner = ExperimentRunner(
            storage=MinIOStorage(...),
            tracker=MLflowServerTracker(...),
            publisher=RedisPublisher(...),
        )
        runner.run(model_class=MyModel, config={"epochs": 10})
    """
    
    def __init__(
        self,
        config: Optional[ConfigInterface] = None,
        storage: Optional[StorageInterface] = None,
        tracker: Optional[TrackerInterface] = None,
        publisher: Optional[PublisherInterface] = None,
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration provider (default: YamlConfig)
            storage: Storage backend (default: LocalStorage)
            tracker: Experiment tracker (default: NoOpTracker, or LocalTracker if MLflow installed)
            publisher: Real-time publisher (default: ConsolePublisher)
        """
        self.config = config or YamlConfig()
        self.storage = storage or LocalStorage(self.config.artifacts_dir)
        
        # Use NoOpTracker by default - it works without MLflow
        if tracker is None:
            from kladml.backends.local_tracker import NoOpTracker
            tracker = NoOpTracker()
        self.tracker = tracker
        
        self.publisher = publisher or ConsolePublisher()
    
    def run(
        self,
        model_class: Type[Any],
        train_data: Any,
        val_data: Optional[Any] = None,
        experiment_name: str = "default",
        run_name: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run an experiment.
        
        Args:
            model_class: Class of the model to train (must have train/evaluate methods)
            train_data: Training data
            val_data: Optional validation data
            experiment_name: Name for the experiment
            run_name: Optional name for this run
            model_config: Configuration passed to model constructor
            training_config: Configuration passed to model.train()
        
        Returns:
            Dictionary with run results including run_id and metrics
        """
        model_config = model_config or {}
        training_config = training_config or {}
        
        # Start tracking
        run_id = self.tracker.start_run(
            experiment_name=experiment_name,
            run_name=run_name,
        )
        
        self.publisher.publish_status(run_id, "RUNNING", "Starting experiment")
        
        try:
            # Log parameters
            self.tracker.log_params({
                "model_class": model_class.__name__,
                **model_config,
                **{f"train_{k}": v for k, v in training_config.items()},
            })
            
            # Instantiate model
            model = model_class(config=model_config)
            
            # Inject publisher for real-time updates
            if hasattr(model, 'set_publisher'):
                model.set_publisher(self.publisher, run_id)
            
            # Train
            self.publisher.publish_status(run_id, "RUNNING", "Training started")
            train_result = model.train(
                X_train=train_data,
                y_train=None,  # Depends on data structure
                **training_config,
            )
            
            # Log training metrics
            if isinstance(train_result, dict):
                self.tracker.log_metrics(train_result)
            
            # Evaluate if validation data provided
            eval_result = {}
            if val_data is not None and hasattr(model, 'evaluate'):
                self.publisher.publish_status(run_id, "RUNNING", "Evaluating")
                eval_result = model.evaluate(X_test=val_data)
                if isinstance(eval_result, dict):
                    self.tracker.log_metrics({f"val_{k}": v for k, v in eval_result.items()})
            
            # Save model
            if hasattr(model, 'save'):
                model_path = Path(self.config.artifacts_dir) / "model"
                model_path.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))
                self.tracker.log_artifact(str(model_path))
            
            self.publisher.publish_status(run_id, "COMPLETED", "Experiment finished successfully")
            self.tracker.end_run("FINISHED")
            
            return {
                "run_id": run_id,
                "status": "COMPLETED",
                "train_metrics": train_result,
                "eval_metrics": eval_result,
            }
            
        except Exception as e:
            self.publisher.publish_status(run_id, "FAILED", str(e))
            self.tracker.end_run("FAILED")
            raise
