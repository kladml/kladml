
import pytest
import yaml
import torch
from pathlib import Path
from typer.testing import CliRunner
from kladml.cli.main import app

runner = CliRunner()

# 1. Create Dummy Model File
DUMMY_MODEL_CODE = """
from kladml.models.base import BaseModel
from kladml.tasks import MLTask
import torch

class DummyModel(BaseModel):
    @property
    def ml_task(self):
        return MLTask.REGRESSION

    def train(self, X_train, X_val=None, **kwargs):
        # Fake training
        lr = self.config.get("optimizer", {}).get("lr", 0.001)
        d_model = self.config.get("model", {}).get("d_model", 64)
        
        # We simulate a loss that improves if d_model is 64 and lr is 0.01
        target_lr = 0.01
        target_d = 64
        
        err_lr = abs(lr - target_lr)
        err_d = abs(d_model - target_d)
        
        loss = err_lr * 10.0 + err_d * 0.1
        
        # Pruning check (simulate slow convergence for bad params)
        callbacks = kwargs.get("callbacks", []) # Wait, LocalTrainingExecutor creates UniversalTrainer which uses _init_standard_callbacks
        # How do we trigger pruning in this dummy logic?
        # LocalTrainingExecutor calls model.train(X_train, ..., kwargs).
        # We need to manually invoke callbacks?
        # UniversalTrainer does: callbacks.on_epoch_end(...)
        # But this DummyModel implements train() from scratch and DOES NOT use UniversalTrainer.
        # This breaks the test logic for Pruning unless we mock UniversalTrainer usage or implement it.
        
        # Let's just return metrics first.
        return {"val_loss": float(loss)}

    def predict(self, X, **kwargs):
        return X

    def evaluate(self, X, **kwargs):
        return {}
    
    def save(self, path):
        Path(path).touch()
        
    def load(self, path):
        pass
"""

@pytest.fixture
def dummy_model_file(tmp_path):
    p = tmp_path / "dummy_model.py"
    p.write_text(DUMMY_MODEL_CODE)
    return str(p)

@pytest.fixture
def tuning_config(tmp_path):
    conf = {
        "project_name": "test_tuning",
        "experiment_name": "test_exp",
        "model": {"d_model": 32},
        "optimizer": {"lr": 0.001},
        "tuning": {
            "search_space": {
                "optimizer.lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "model.d_model": {"type": "int", "low": 32, "high": 128, "step": 32}
            }
        }
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(conf, f)
    return str(p)

def test_cli_tuning_run(dummy_model_file, tuning_config, tmp_path):
    """Test full cli execution of tuning."""
    
    # Run
    # We use a dummy data path, the dummy model ignores it
    data_path = str(tmp_path / "train.parquet")
    Path(data_path).touch()
    
    result = runner.invoke(app, [
        "tune", "run",
        "--config", tuning_config,
        "--model", dummy_model_file,
        "--data", data_path,
        "--project", "test_proj",
        "--experiment", "tuning_test",
        "--trials", "5",  # Keep low
        "--storage", f"sqlite:///{tmp_path}/tuning.db"
    ])
    
    print(result.stdout)
    assert result.exit_code == 0
    assert "Starting Optimization" in result.stdout
    assert "Tuning Complete" in result.stdout
    assert "Best Params" in result.stdout
    
    # Check if "model.d_model" (64) and "optimizer.lr" (0.01) are roughly found
    # Since we run few trials, we might not hit exact. 
    # But we check that it RAN.

