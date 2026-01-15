"""
Functional tests for CLI flow.
"""

import pytest
import os
import shutil
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from kladml.cli.main import app
from kladml.db.session import reset_db, init_db

# Mock model content for training test
DUMMY_MODEL_CONTENT = """
from kladml.base import BaseArchitecture
from kladml.tasks import MLTask

class TestModel(BaseArchitecture):
    def __init__(self, **kwargs):
        super().__init__()
    
    @property
    def ml_task(self) -> MLTask:
        return MLTask.CLASSIFICATION

    def train(self, X_train=None, y_train=None, **kwargs):
        return {"loss": 0.5}
        
    def predict(self, X, **kwargs): return []
    def evaluate(self, X, y=None, **kwargs): return {}
    def save(self, path): pass
    def load(self, path): pass
"""

@pytest.fixture(scope="module")
def runner():
    return CliRunner(env={"NO_COLOR": "1"})

@pytest.fixture(autouse=True)
def setup_cli_env():
    """Setup temp workspace for CLI tests."""
    # Force reset globals in session module just in case
    import kladml.db.session
    kladml.db.session._engine = None
    kladml.db.session._session_factory = None

    # Create temp dir for execution
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    
    # Setup DB path
    db_path = Path(temp_dir) / "test_cli.db"
    os.environ["KLADML_DB_PATH"] = str(db_path)
    
    # Setup MLflow path to avoid polluting user's ./mlruns
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{temp_dir}/mlflow.db"
    
    # Reset DB
    try:
        reset_db()
        init_db()
    except:
        pass
        
    yield temp_dir
    
    # Cleanup
    os.chdir(cwd)
    shutil.rmtree(temp_dir)
    if "KLADML_DB_PATH" in os.environ:
        del os.environ["KLADML_DB_PATH"]


def test_project_lifecycle(runner, setup_cli_env):
    """Test project creation and listing."""
    # Create
    result = runner.invoke(app, ["project", "create", "cli-test-proj"])
    assert result.exit_code == 0
    assert "Created project 'cli-test-proj'" in result.stdout
    
    # List
    result = runner.invoke(app, ["project", "list"])
    assert result.exit_code == 0
    assert "cli-test-proj" in result.stdout
    
    # Show
    result = runner.invoke(app, ["project", "show", "cli-test-proj"])
    assert result.exit_code == 0
    assert "cli-test-proj" in result.stdout


def test_experiment_lifecycle(runner, setup_cli_env):
    """Test experiment creation and listing."""
    # Setup project
    runner.invoke(app, ["project", "create", "exp-proj"])
    
    # Create experiment
    result = runner.invoke(app, ["experiment", "create", "-p", "exp-proj", "-n", "exp-1"])
    assert result.exit_code == 0
    assert "Created experiment 'exp-1'" in result.stdout
    
    # List experiments
    result = runner.invoke(app, ["experiment", "list", "-p", "exp-proj"])
    assert result.exit_code == 0
    assert "exp-1" in result.stdout
    
    # Verify MLflow integration (via list output showing status/ID)
    assert "active" in result.stdout


def test_train_single_flow(runner, setup_cli_env):
    """Test single training command flow."""
    # Setup files
    with open("model.py", "w") as f:
        f.write(DUMMY_MODEL_CONTENT)
    Path("data.csv").touch()
    
    # Run training
    # Note: This runs LocalTrainingExecutor -> LocalTracker -> MLflow
    # It creates project/experiment implicitly if missing
    result = runner.invoke(app, [
        "train", "single",
        "model.py",
        "--data", "data.csv",
        "-p", "train-proj",
        "-e", "train-exp"
    ])
    
    assert result.exit_code == 0
    assert "Training complete" in result.stdout
    assert "Metrics" in result.stdout
    assert "loss" in result.stdout
    
    # Verify project/experiment created
    result = runner.invoke(app, ["experiment", "list", "-p", "train-proj"])
    assert "train-exp" in result.stdout
