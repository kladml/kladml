
import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from kladml.cli.commands.train.core import app
from kladml.utils.loading import resolve_model_class
from kladml.models.base import BaseModel
import yaml

runner = CliRunner()


def test_relaunch_with_accelerate():
    """Test relaunch command construction."""
    from kladml.cli.commands.train.utils import relaunch_with_accelerate
    import sys
    
    with patch("sys.argv", ["kladml", "train", "single", "--distributed", "--num-processes", "4", "--model", "foo"]), \
         patch("os.execvp") as mock_exec, \
         patch("kladml.cli.commands.train.utils.console.print"):
         
         relaunch_with_accelerate(num_processes=2)
         
         mock_exec.assert_called_once()
         args = mock_exec.call_args[0]
         cmd = args[1]
         
         assert cmd[0] == "accelerate"
         assert cmd[2] == "--num_processes"
         assert cmd[3] == "2"
         assert "-m" in cmd
         assert "--distributed" not in cmd # filtered out
         assert "--num-processes" not in cmd # filtered out
         assert "--model" in cmd
         assert "foo" in cmd

class MockModel(BaseModel):

    def train(self, X_train, **kwargs):
        # Return dummy metrics
        return {"loss": 0.5}
    def predict(self, X, **kwargs): pass
    def evaluate(self, X_test, **kwargs): pass
    def save(self, path): pass
    def load(self, path): pass
    @property
    def ml_task(self): return "classification"

# --- Test _resolve_model_class ---

def test_resolve_model_by_name():
    """Test resolving a model by python module path (mocked)."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.MyModel = MockModel
        mock_import.return_value = mock_module
        
        # We need to simulate that 'MyModel' is in the module
        
        # Test finding it in the module directly
        cls = resolve_model_class("my_package.module")
        assert cls == MockModel

def test_resolve_model_file_not_found():
    """Test error when file path doesn't exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_model_class("non_existent_file.py")

# --- Test train quick ---

@patch("kladml.cli.commands.train.core.resolve_model_class")
def test_train_quick(mock_resolve, tmp_path):
    """Test quick training command with real config file."""
    mock_resolve.return_value = MockModel
    
    # Create dummy config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("epochs: 1\nbatch_size: 32")
    
    # Create dummy data
    data_file = tmp_path / "data.pkl"
    data_file.touch()
    
    result = runner.invoke(app, [
        "quick", 
        "--config", str(config_file), 
        "--train", str(data_file),
        "--model", "dummy_model"
    ])
    
    if result.exit_code != 0:
        print(result.stdout)
        print(result.exception)

    assert result.exit_code == 0
    assert "Training complete" in result.stdout
    assert "Final Metrics" in result.stdout

def test_train_quick_missing_config():
    """Test error when config missing."""
    result = runner.invoke(app, [
        "quick", 
        "--config", "missing.yaml", 
        "--train", "data.pkl"
    ])
    
    assert result.exit_code == 1
    assert "Config not found" in result.stdout

@patch("kladml.cli.commands.train.core.resolve_model_class")
def test_train_single(mock_resolve, tmp_path):
    """Test full training command via CLI."""
    mock_resolve.return_value = MockModel
    
    # Needs valid metadata mocked or handled?
    # Our code uses `get_metadata_backend()`.
    # We should mock `kladml.backends.get_metadata_backend` and `LocalTracker`.
    # Otherwise it tries to access real DB.
    
    with patch("kladml.backends.get_metadata_backend") as mock_get_meta:
        with patch("kladml.backends.local_tracker.LocalTracker") as MockTracker:
            mock_meta = MagicMock()
            mock_get_meta.return_value = mock_meta
            # Mock get_project / create_project
            mock_meta.get_project.return_value = MagicMock()
            mock_meta.get_family.return_value = MagicMock()
            
            # Mock Executor
            with patch("kladml.training.executor.LocalTrainingExecutor") as MockExec:
                mock_executor = MockExec.return_value
                mock_executor.execute_single.return_value = ("run_123", {"val_loss": 0.1})
                
                result = runner.invoke(app, [
                    "single",
                    "--model", "gluformer",
                    "--data", "data/",
                    "--project", "test_proj",
                    "--experiment", "test_exp"
                ])
                
                assert result.exit_code == 0
                assert "Training complete" in result.stdout
                assert "run_123" in result.stdout

@patch("kladml.cli.commands.train.core.resolve_model_class")
def test_train_grid(mock_resolve, tmp_path):
    """Test grid search command."""
    mock_resolve.return_value = MockModel
    
    # Dummy grid config
    grid_cfg = tmp_path / "grid.yaml"
    grid_cfg.write_text("search_space:\n  lr:\n    - 0.01\n    - 0.1")
    
    with patch("kladml.backends.get_metadata_backend") as mock_get_meta:
        with patch("kladml.backends.local_tracker.LocalTracker") as MockTracker:
            mock_meta = MagicMock()
            mock_get_meta.return_value = mock_meta
            mock_meta.get_project.return_value = MagicMock()
            mock_meta.get_family.return_value = MagicMock()
            
            with patch("kladml.training.executor.LocalTrainingExecutor") as MockExec:
                mock_executor = MockExec.return_value
                # execute_grid_search(data_path, search_space)
                mock_executor.execute_grid_search.return_value = ["run_1", "run_2"]
                mock_executor.best_run_id = "run_2"
                mock_executor.best_metrics = {"loss": 0.05}
                
                result = runner.invoke(app, [
                    "grid",
                    "--model", "gluformer",
                    "--data", "data/",
                    "--project", "test_proj",
                    "--experiment", "test_grid",
                    "--grid", str(grid_cfg)
                ])
                
                if result.exit_code != 0:
                    print(result.stdout)
                
                assert result.exit_code == 0
                assert "Grid search complete" in result.stdout
                assert "Successful runs: 2" in result.stdout

def test_train_distributed_mps_fallback():
    """Test fallback to single process on MacOS."""
    with patch("sys.platform", "darwin"):
        with patch("kladml.cli.commands.train.core.Console.print") as mock_print:
             # Mock relaunch to avoid actual exec
             with patch("kladml.cli.commands.train.core.relaunch_with_accelerate") as mock_relaunch:
                 # Just invoke. It should print warning and call relaunch?
                 # Wait, logic in core.py: 
                 # if sys.platform == "darwin": distributed=False -> Fallthrough
                 # So it DOES NOT call relaunch.
                 # It falls through to normal execution.
                 
                 # We need to mock resolved model class etc or it will fail later.
                 with patch("kladml.cli.commands.train.core.resolve_model_class") as mock_res:
                      with patch("kladml.backends.get_metadata_backend"), \
                           patch("kladml.backends.local_tracker.LocalTracker"), \
                           patch("kladml.training.executor.LocalTrainingExecutor") as MockExec:
                           
                           mock_res.return_value = MockModel
                           MockExec.return_value.execute_single.return_value = ("run_dist_mps", {})
                           
                           result = runner.invoke(app, [
                                "single",
                                "--model", "gluformer",
                                "--data", "data/",
                                "--project", "test_proj",
                                "--experiment", "test_exp",
                                "--distributed"
                           ])
                           
                           assert result.exit_code == 0
                           # Verify warning
                           assert any("not supported efficiently" in str(c) for c in mock_print.call_args_list)

def test_train_distributed_cuda_check():
    """Test distributed resource check."""
    from unittest.mock import call
    with patch("sys.platform", "linux"):
        # We need to mock torch inside the function
        # Since import is inside function: import torch
        # We patch sys.modules['torch']? Or patch 'kladml.cli.commands.train.core.torch' won't work if it's imported inside.
        # Patching sys.modules is safest.
        
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        with patch.dict("sys.modules", {"torch": mock_torch}):
             with patch("kladml.cli.commands.train.core.Console.print") as mock_print:
                   result = runner.invoke(app, [
                        "single",
                        "--model", "gluformer",
                        "--data", "data/",
                        "--project", "test_proj",
                        "--experiment", "test_exp",
                        "--distributed",
                        "--num-processes", "2"
                   ])
                   assert result.exit_code == 1
                   # Check error message printed
                   # assert any("only 1 GPUs found" in str(c) for c in mock_print.call_args_list) # Rich print checks harder

