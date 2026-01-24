
import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from kladml.cli.train import app, _resolve_model_class
from kladml.models.base import BaseModel
import yaml

runner = CliRunner()

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
        cls = _resolve_model_class("my_package.module")
        assert cls == MockModel

def test_resolve_model_file_not_found():
    """Test error when file path doesn't exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_model_class("non_existent_file.py")

# --- Test train quick ---

@patch("kladml.cli.train._resolve_model_class")
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
