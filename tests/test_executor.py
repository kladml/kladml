"""
Unit tests for LocalTrainingExecutor.
"""

import pytest
import shutil
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from unittest.mock import MagicMock, ANY

from kladml.training.executor import LocalTrainingExecutor
from kladml.models.base import BaseModel
from kladml.interfaces import TrackerInterface

from kladml.tasks import MLTask

class DummyModel(BaseModel):
    """Dummy model for testing."""
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        
    @property
    def ml_task(self) -> MLTask:
        return MLTask.CLASSIFICATION
        
    def train(self, X_train=None, y_train=None, **kwargs):
        # Return dummy metrics
        return {"loss": 0.1, "accuracy": 0.9}
        
    def predict(self, X, **kwargs):
        return [0] * len(X)
        
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"loss": 0.1}
        
    def save(self, path: str):
        pass
        
    def load(self, path: str):
        pass


class MockTracker(TrackerInterface):
    """Mock tracker for testing executor interactions."""
    def __init__(self):
        self.runs = []
        self.params = {}
        self.metrics = {}
        self._active_run_id = None
        
    def start_run(self, experiment_name, run_name=None, tags=None):
        self._active_run_id = f"mock-run-{len(self.runs)}"
        self.runs.append({
            "id": self._active_run_id,
            "experiment": experiment_name,
            "name": run_name
        })
        return self._active_run_id
        
    def end_run(self, status="FINISHED"):
        self._active_run_id = None
        
    def log_param(self, key, value):
        self.params[key] = value
        
    def log_params(self, params):
        self.params.update(params)
        
    def log_metric(self, key, value, step=None):
        self.metrics[key] = value
        
    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)
        
    def log_artifact(self, local_path, artifact_path=None): pass
    def log_model(self, model, artifact_path, **kwargs): pass
    
    @property
    def active_run_id(self): return self._active_run_id
    def get_artifact_uri(self, artifact_path=""): return ""
    
    # Management stubs needed for init compatibility if used
    def search_experiments(self, filter_string=None): return []
    def get_experiment_by_name(self, name): return None
    def create_experiment(self, name): return "mock-exp-id"
    def search_runs(self, experiment_id, **kwargs): return []
    def get_run(self, run_id): return None


@pytest.fixture
def executor():
    tracker = MockTracker()
    return LocalTrainingExecutor(
        model_class=DummyModel,
        experiment_name="test-exp",
        tracker=tracker,
        config={"base_param": 10}
    ), tracker


def test_execute_single(executor, caplog):
    """Test single execution."""
    exec_inst, tracker = executor
    caplog.set_level(logging.DEBUG)
    
    run_id, metrics = exec_inst.execute_single(
        data_path="dummy/path",
        params={"lr": 0.01},
        run_name="single-test"
    )
    
    if run_id is None:
        print("\nCaptured Logs:")
        print(caplog.text)
    
    assert run_id is not None
    assert metrics == {"loss": 0.1, "accuracy": 0.9}
    
    # Verify tracker calls
    assert len(tracker.runs) == 1
    assert tracker.runs[0]["name"] == "single-test"
    assert tracker.params["lr"] == 0.01
    assert tracker.params["base_param"] == 10  # Merged config
    assert tracker.metrics["accuracy"] == 0.9


def test_execute_grid_search(executor):
    """Test grid search execution."""
    exec_inst, tracker = executor
    
    search_space = {
        "lr": [0.01, 0.001],
        "batch_size": [32]
    }
    
    run_ids = exec_inst.execute_grid_search(
        data_path="dummy/path",
        search_space=search_space
    )
    
    # Should create 2 runs (2 lr * 1 bs)
    assert len(run_ids) == 2
    assert len(tracker.runs) == 2
    
    # Check best model tracking
    assert exec_inst.best_run_id is not None
    assert exec_inst.best_metrics is not None

def test_semantic_run_naming(executor):
    """Test semantic name generation."""
    exec_inst, _ = executor
    
    params = {"learning_rate": 0.001, "epochs": 50, "other": "val"}
    name = exec_inst._semantic_run_name(params, 1, 10)
    
    assert "DummyModel" in name
    assert "lr1e-03" in name  # Abbreviation + float fmt
    assert "ep50" in name     # Abbreviation
    assert "othval" in name   # other -> oth, value -> val
    assert "1of10" in name
