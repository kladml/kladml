"""
Training Executor for KladML SDK.

Provides local training execution with:
- Grid search support
- Parameter combination generation
- Best model tracking
- MLflow integration (no local DB for runs)
"""

import itertools
import time
import logging
import uuid
import os
from typing import Dict, Any, List, Optional, Tuple, Type
from pathlib import Path

from kladml.models.base import BaseModel
from kladml.interfaces.tracker import TrackerInterface
from kladml.interfaces.publisher import PublisherInterface
from kladml.utils.paths import resolve_dataset_path, resolve_preprocessor_path

logger = logging.getLogger(__name__)


class LocalTrainingExecutor:
    """
    Local training executor with grid search support.
    
    Executes model training runs locally, tracking results
    directly in MLflow (no separate SQLite for runs).
    
    Features:
        - Grid search over parameter combinations
        - Semantic run naming
        - Best model tracking
        - MLflow integration
        - Automatic path resolution for datasets/preprocessors
    
    Example:
        >>> executor = LocalTrainingExecutor(
        ...     model_class=MyModel,
        ...     experiment_name="baseline",
        ...     config={"epochs": 10}
        ... )
        >>> run_ids = executor.execute_grid_search(
        ...     data_path="my_dataset",  # -> data/datasets/my_dataset
        ...     search_space={"lr": [0.01, 0.001]}
        ... )
    """
    
    def __init__(
        self,
        model_class: Type[BaseModel],
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[TrackerInterface] = None,
        publisher: Optional[PublisherInterface] = None,
    ):
        """
        Initialize the executor.
        
        Args:
            model_class: The model class to instantiate and train
            experiment_name: Name of the MLflow experiment
            config: Base configuration for training
            tracker: Optional tracker for MLflow integration
            publisher: Optional publisher for real-time events
        """
        self.model_class = model_class
        self.experiment_name = experiment_name
        self.config = config or {}
        self.tracker = tracker
        self.publisher = publisher
        
        # Track best run across grid search
        self._best_run_id: Optional[str] = None
        self._best_metric: Optional[float] = None
        self._best_metrics: Optional[Dict[str, float]] = None
        
        # Comparison settings
        self._comparison_metric = self.config.get("comparison_metric", "loss")
        self._lower_is_better = self.config.get("lower_is_better", True)
    
    def execute_grid_search(
        self,
        data_path: str,
        search_space: Dict[str, List[Any]],
    ) -> List[str]:
        """
        Execute grid search over parameter combinations.
        
        Args:
            data_path: Path to training data (resolved relative to data/datasets/ if not absolute)
            search_space: Dict of param_name -> list of values
            
        Returns:
            List of MLflow run IDs
        """
        resolved_data_path = str(resolve_dataset_path(data_path))
        
        combinations = self._generate_combinations(search_space)
        
        if not combinations:
            combinations = [{}]
        
        logger.info(f"Grid Search: {len(combinations)} combinations")
        logger.info(f"Using dataset: {resolved_data_path}")
        
        # Reset best tracking
        self._best_run_id = None
        self._best_metric = None
        self._best_metrics = None
        
        run_ids: List[str] = []
        
        for i, params in enumerate(combinations, 1):
            run_name = self._semantic_run_name(params, i, len(combinations))
            
            logger.info(f"Run {i}/{len(combinations)}: {run_name}")
            logger.debug(f"Params: {params}")
            
            try:
                run_id, metrics = self._execute_single_run(
                    data_path=resolved_data_path,
                    params=params,
                    run_name=run_name,
                )
                
                if run_id:
                    run_ids.append(run_id)
                    self._update_best(run_id, metrics)
                    logger.info(f"Run {i} completed: {run_id[:8]}")
                    
            except Exception as e:
                logger.error(f"Run {i} failed: {e}")
            
            # Brief pause between runs to allow cleanup
            if i < len(combinations):
                time.sleep(1)
        
        logger.info(f"Grid Search complete: {len(run_ids)}/{len(combinations)} successful")
        
        if self._best_run_id:
            logger.info(f"Best run: {self._best_run_id[:8]} "
                       f"({self._comparison_metric}={self._best_metric:.4f})")
        
        return run_ids
    
    def execute_single(
        self,
        data_path: str,
        val_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """
        Execute a single training run.
        
        Args:
            data_path: Path to training data (resolved relative to data/datasets/ if not absolute)
            val_path: Optional path to validation data
            params: Parameters for this run
            run_name: Optional custom run name
            
        Returns:
            Tuple of (run_id, metrics) or (None, None) if failed
        """
        params = params or {}
        run_name = run_name or self._semantic_run_name(params, 1, 1)
        
        run_name = run_name or self._semantic_run_name(params, 1, 1)
        
        resolved_data_path = str(resolve_dataset_path(data_path))
        resolved_val_path = str(resolve_dataset_path(val_path)) if val_path else None
        
        # Also resolve preprocessor if present in params/config
        # Note: params override config
        merged_config = {**self.config, **params}
        if "preprocessor" in merged_config:
             prep_path = resolve_preprocessor_path(merged_config["preprocessor"])
             params["preprocessor"] = str(prep_path)
             # Update merged config implicitly in _execute_single_run but good to have explicit resolve logic here if needed
        
             # Update merged config implicitly in _execute_single_run but good to have explicit resolve logic here if needed
        
        return self._execute_single_run(resolved_data_path, params, run_name, val_path=resolved_val_path)
    
    def _execute_single_run(
        self,
        data_path: str,
        params: Dict[str, Any],
        run_name: str,
        val_path: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """
        Internal method to execute a single run.
        
        All run tracking is done via MLflow, no local database.
        """
        # Merge base config with run-specific params
        run_config = {**self.config, **params}
        run_config["data_path"] = data_path
        if val_path:
            run_config["val_path"] = val_path
        
        # Resolve 'preprocessor' in config if likely a file path
        if "preprocessor" in run_config:
             run_config["preprocessor"] = str(resolve_preprocessor_path(run_config["preprocessor"]))
        
        # Generate a local run ID (will be replaced by MLflow run ID if available)
        local_run_id = str(uuid.uuid4())[:8]
        run_id = local_run_id
        
        # Publish start event
        if self.publisher:
            self.publisher.publish("run_start", {
                "run_id": run_id,
                "run_name": run_name,
                "params": params,
            })
        
        # Start MLflow run if tracker available
        if self.tracker:
            run_id = self.tracker.start_run(
                experiment_name=self.experiment_name,
                run_name=run_name,
            )
            # Log all config params (merged)
            # Filter out complex objects like data_path if it's not a string/number
            loggable_params = {
                k: v for k, v in run_config.items() 
                if isinstance(v, (str, int, float, bool))
            }
            self.tracker.log_params(loggable_params)
        
        try:
            # Instantiate and train model
            # BaseModel expects a 'config' dictionary
            model = self.model_class(config=run_config)
            metrics = model.train(X_train=data_path, X_val=val_path, **run_config)
            
            # Ensure metrics is a dict
            if not isinstance(metrics, dict):
                metrics = {"result": metrics} if metrics else {}
            
            # Log metrics to tracker
            if self.tracker:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tracker.log_metric(key, value)
            
            # Publish complete event
            if self.publisher:
                self.publisher.publish("run_complete", {
                    "run_id": run_id,
                    "metrics": metrics,
                })
            
            # End MLflow run
            if self.tracker:
                self.tracker.end_run(status="FINISHED")
            
            return run_id, metrics
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            
            # Publish error event
            if self.publisher:
                self.publisher.publish("run_error", {
                    "run_id": run_id,
                    "error": str(e),
                })
            
            # End MLflow run with failure
            if self.tracker:
                self.tracker.end_run(status="FAILED")
            
            return None, None
    
    def _update_best(
        self,
        run_id: str,
        metrics: Optional[Dict[str, float]],
    ) -> None:
        """Update best run tracking."""
        if not metrics:
            return
        
        current_value = metrics.get(self._comparison_metric)
        if current_value is None:
            return
        
        if self._best_metric is None:
            self._best_run_id = run_id
            self._best_metric = current_value
            self._best_metrics = metrics
            logger.debug(f"First run - setting as best "
                        f"({self._comparison_metric}={current_value:.4f})")
        else:
            is_better = (
                (current_value < self._best_metric) if self._lower_is_better
                else (current_value > self._best_metric)
            )
            if is_better:
                logger.info(f"New best! {self._comparison_metric}={current_value:.4f} "
                           f"(was {self._best_metric:.4f})")
                self._best_run_id = run_id
                self._best_metric = current_value
                self._best_metrics = metrics
    
    def _semantic_run_name(
        self,
        params: Dict[str, Any],
        trial: int,
        total: int,
    ) -> str:
        """
        Generate semantic run name.
        
        Returns:
            Semantic name like "Model_lr0.01_ep10_1of8"
        """
        model_name = self.model_class.__name__
        
        # Format key params for name
        param_parts = []
        for key, value in sorted(params.items()):
            # Abbreviate common param names
            abbrev = {
                "learning_rate": "lr",
                "hidden_size": "h",
                "num_layers": "nl",
                "epochs": "ep",
                "batch_size": "bs",
            }.get(key, key[:3])
            
            # Format value
            if isinstance(value, float):
                param_parts.append(f"{abbrev}{value:.0e}")
            else:
                param_parts.append(f"{abbrev}{value}")
        
        params_str = "_".join(param_parts) if param_parts else "default"
        
        return f"{model_name}_{params_str}_{trial}of{total}"
    
    @staticmethod
    def _generate_combinations(
        search_space: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        if not search_space:
            return [{}]
        
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    @property
    def best_run_id(self) -> Optional[str]:
        """Get the ID of the best run from the last grid search."""
        return self._best_run_id
    
    @property
    def best_metrics(self) -> Optional[Dict[str, float]]:
        """Get the metrics of the best run from the last grid search."""
        return self._best_metrics
