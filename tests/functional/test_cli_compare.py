
import pytest
from typer.testing import CliRunner
from kladml.cli.main import app
from kladml.backends.local_tracker import LocalTracker
import tempfile
import os
import shutil
from pathlib import Path
from kladml.db.session import reset_db, init_db
from kladml.config.settings import settings

@pytest.fixture
def runner():
    return CliRunner(env={"NO_COLOR": "1"})

@pytest.fixture
def setup_compare_env():
    """Setup environment with populated runs for comparison."""
    # Reset globals
    import kladml.db.session
    kladml.db.session._engine = None
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # DB Setup
    db_path = Path(temp_dir) / "compare_test.db"
    db_uri = f"sqlite:///{db_path}"
    
    # Patch settings
    os.environ["KLADML_DB_PATH"] = str(db_path)
    # MLflow URI patch for LocalTracker inside CLI
    orig_uri = settings.mlflow_tracking_uri
    settings.mlflow_tracking_uri = db_uri
    
    # Init DB
    init_db()
    
    # Initialize Tracker and populate data
    tracker = LocalTracker(tracking_dir=temp_dir)
    
    # Create Experiment
    exp_id = tracker.create_experiment("compare-exp")
    
    # Run 1
    run1_id = tracker.start_run("compare-exp", run_name="run-v1")
    tracker.log_metric("val_loss", 0.5)
    tracker.log_metric("accuracy", 0.8)
    tracker.log_param("model_type", "version_1")
    tracker.end_run()
    
    # Run 2
    run2_id = tracker.start_run("compare-exp", run_name="run-v2")
    tracker.log_metric("val_loss", 0.3)  # Better
    tracker.log_metric("accuracy", 0.9)
    tracker.log_param("model_type", "version_2")
    tracker.end_run()
    
    yield [run1_id, run2_id]
    
    # Cleanup
    settings.mlflow_tracking_uri = orig_uri
    os.chdir(cwd)
    shutil.rmtree(temp_dir)
    if "KLADML_DB_PATH" in os.environ:
        del os.environ["KLADML_DB_PATH"]

def test_compare_runs_cli(runner, setup_compare_env):
    """Test kladml compare command."""
    run_ids = setup_compare_env
    run1, run2 = run_ids
    
    # Convert IDs to comma-separated string
    runs_arg = f"{run1},{run2}"
    
    # Run compare command (assuming it will be: kladml compare --runs ID1,ID2)
    result = runner.invoke(app, ["compare", "--runs", runs_arg])
    
    assert result.exit_code == 0
    assert "Comparison" in result.stdout
    
    # Verify metrics appear
    assert "val_loss" in result.stdout
    assert "0.5" in result.stdout
    assert "0.3" in result.stdout
    
    # Verify params appear
    assert "model_type" in result.stdout
    assert "version_1" in result.stdout
