"""
Unit tests for LocalTracker.
"""

import pytest
import shutil
import tempfile
from pathlib import Path
from kladml.backends.local_tracker import LocalTracker
from kladml.config.settings import settings

@pytest.fixture
def tracker():
    """Create a tracker with a temporary database."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_tracker.db"
    
    # Patch settings to use isolated DB
    original_uri = settings.mlflow_tracking_uri
    settings.mlflow_tracking_uri = f"sqlite:///{db_path}"
    
    tracker = LocalTracker(tracking_dir=temp_dir)
    yield tracker
    
    # Cleanup
    settings.mlflow_tracking_uri = original_uri
    shutil.rmtree(temp_dir)

def test_tracker_initialization(tracker):
    """Test tracker initialization."""
    assert Path(tracker.tracking_dir).exists()
    assert tracker._active_run is None

def test_experiment_management(tracker):
    """Test creating and finding experiments."""
    # Create
    exp_id = tracker.create_experiment("test-exp")
    assert exp_id is not None
    
    # Get by name
    exp = tracker.get_experiment_by_name("test-exp")
    assert exp is not None
    assert exp["id"] == exp_id
    assert exp["name"] == "test-exp"
    
    # Search
    exps = tracker.search_experiments()
    assert len(exps) >= 1
    names = [e["name"] for e in exps]
    assert "test-exp" in names

def test_run_lifecycle(tracker):
    """Test start, log, and end run."""
    tracker.create_experiment("lifecycle-exp")
    
    # Start run
    run_id = tracker.start_run("lifecycle-exp", run_name="test-run")
    assert run_id is not None
    assert tracker.active_run_id == run_id
    
    # Log params and metrics
    tracker.log_param("learning_rate", 0.01)
    tracker.log_metric("accuracy", 0.95)
    
    # End run
    tracker.end_run()
    assert tracker.active_run_id is None
    
    # Verify logged data via get_run
    run = tracker.get_run(run_id)
    assert run["status"] == "FINISHED"
    assert run["params"]["learning_rate"] == "0.01"  # MLflow stores params as strings
    assert run["metrics"]["accuracy"] == 0.95

def test_search_runs(tracker):
    """Test searching runs."""
    tracker.create_experiment("search-exp")
    exp = tracker.get_experiment_by_name("search-exp")
    
    # Create 2 runs
    tracker.start_run("search-exp", run_name="run1")
    tracker.log_metric("loss", 0.5)
    tracker.end_run()
    
    tracker.start_run("search-exp", run_name="run2")
    tracker.log_metric("loss", 0.3)
    tracker.end_run()
    
    # Search all
    runs = tracker.search_runs(exp["id"])
    assert len(runs) == 2
    
    # Check if metrics are retrieved
    losses = [r["metrics"]["loss"] for r in runs]
    assert 0.5 in losses
    assert 0.3 in losses
