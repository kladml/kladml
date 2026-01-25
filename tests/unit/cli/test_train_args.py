
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.commands.train.core import app

runner = CliRunner()

@pytest.fixture
def mock_trainer():
    with patch("kladml.training.executor.LocalTrainingExecutor") as mock_t:
        yield mock_t

def test_train_arg_parsing(mock_trainer, tmp_path):
    # Create dummy config
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  name: gluformer\n  d_model: 64")
    
    # Mock model loading to avoid import errors
    with patch("kladml.cli.commands.train.core.resolve_model_class") as mock_resolve, \
         patch("kladml.backends.get_metadata_backend") as mock_meta:

        mock_class = MagicMock()
        mock_class.__name__ = "MockModel"
        mock_resolve.return_value = mock_class
        mock_meta.return_value = MagicMock()
        
        # Configure executor instance
        mock_executor_instance = mock_trainer.return_value
        mock_executor_instance.execute_single.return_value = ("run-123", {"loss": 0.1})
        
        # Invoke 'single' command with required args
        result = runner.invoke(app, [
            "single",
            "--config", str(cfg),
            "--model", "gluformer",
            "--data", "train.pkl",
            "--project", "test-proj", 
            "--experiment", "exp-1",
            # removed invalid args like --epochs
        ])
        
        assert result.exit_code == 0, f"Stdout: {result.stdout}"
        
        # Verify overrides merged
        mock_trainer.assert_called_once()
        # Check config in init args
        call_kwargs = mock_trainer.call_args[1]
        assert "config" in call_kwargs
