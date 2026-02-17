
"""
Unit tests for LocalTracker.
"""

import pytest
import shutil
import tempfile
import os
import importlib
from pathlib import Path
from kladml.backends.local_tracker import LocalTracker

@pytest.fixture
def tracker():
    """Create a tracker with a temporary database."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_tracker.db"

    # Set environment variable for this test
    os.environ["KLADML_MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path}"

    # Reload settings to pick up new env var
    import kladml.config.settings
    importlib.reload(kladml.config.settings)

    tracker = LocalTracker(tracking_dir=temp_dir)
    yield tracker

    # Cleanup
    if "KLADML_MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["KLADML_MLFLOW_TRACKING_URI"]
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

def test_log_artifact(tracker):
    """Test logging a local artifact."""
    tracker.create_experiment("artifact-exp")
    tracker.start_run("artifact-exp")
    
    # Create dummy artifact
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        f_path = f.name
        
    try:
        tracker.log_artifact(f_path, "my_artifacts")
        # No exception implies success
    finally:
        os.unlink(f_path)
        tracker.end_run()

def test_log_model(tracker):
    """Test logging a model (pickle)."""
    tracker.create_experiment("model-exp")
    tracker.start_run("model-exp")
    
    model = {"my": "model"}
    tracker.log_model(model, "model_dir")
    
    tracker.end_run()
